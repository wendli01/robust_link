import itertools
import os
import time
import warnings
from typing import Sequence, Union, Optional

import numpy as np
import pandas as pd
import regex
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.base import TransformerMixin


class DocumentSentenceTransformer(TransformerMixin):
    def __init__(self, model_name: str = 'jinaai/jina-embeddings-v2-base-de', cache_fn=None,
                 overlap: int = 1024, verbose=True, chunk_size: Optional[int] = None, timings_file='.cache/timings.csv',
                 aggregate: bool = True, **model_kwargs):
        self.model_name = model_name
        self.model_, self.out_feat_ = None, None
        self.model_kwargs = model_kwargs
        self.cache_fn = cache_fn
        self.using_cached_ = False
        self.overlap = overlap
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.timings_file = timings_file
        self.aggregate = aggregate

    def fit(self, *_, **__):
        if self.cache_fn not in (None, ''):
            if not os.path.exists(self.cache_fn):
                warnings.warn(f'Cache file {self.cache_fn} does not exist, transforming documents from scratch')
            else:
                self.using_cached_ = True
                return self
        self.model_ = SentenceTransformer(self.model_name, trust_remote_code=True, **self.model_kwargs)

        if self.chunk_size is None:
            chunk_size = self.model_.max_seq_length
            if self.verbose:
                print('chunk size', chunk_size)
        else:
            chunk_size = self.chunk_size

        self.text_splitter_ = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=self.overlap)
        self.out_feat_ = self.model_.get_sentence_embedding_dimension()
        return self

    def transform(self, documents: Sequence[str], **model_kwargs) -> np.ndarray:
        if self.using_cached_:
            return np.load(self.cache_fn, allow_pickle=True)

        if self.model_ is None:
            raise RuntimeError('model not fitted')

        patches = [self.text_splitter_.split_text(d) for d in documents]
        lengths = [len(p) for p in patches]
        sentences = list(itertools.chain(*patches))

        self.model_: SentenceTransformer
        start = time.time()
        sentence_embeddings = self.model_.encode(sentences, **model_kwargs)
        embedding_time = time.time() - start
        document_embeddings = np.split(sentence_embeddings, np.cumsum([l for l in lengths])[:-1])

        if self.aggregate:
            document_embeddings = np.array([d.mean(0) for d in document_embeddings])
        else:
            res = document_embeddings
            document_embeddings = np.empty(len(res), object)
            document_embeddings[:] = res

        if self.cache_fn not in (None, ''):
            print(f'Saving document embeddings to {self.cache_fn}')
            np.save(self.cache_fn, document_embeddings)

            if self.timings_file not in (None, ''):
                if os.path.exists(self.timings_file):
                    timings = pd.read_csv(self.timings_file, index_col = 'index')['time'].to_dict()
                else:
                    timings = {}
                timings[self.cache_fn.split('/')[-1].split('.')[0]] = round(embedding_time, 2)
                print(f'saving embedding time {round(embedding_time)}s to {self.timings_file}')
                pd.Series(timings, name='time').to_frame().reset_index().to_csv(self.timings_file, index=False)

        return document_embeddings
