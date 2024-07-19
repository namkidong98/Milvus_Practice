# Milvus_Practice
Vector search example for retrieving related articles in Milvus VectorDB

<br>

## Description
1. milvus_tutorial.ipynb
    - <a href="https://hotorch.tistory.com/415">참고 사이트</a>를 따라 제공된 기사 데이터셋(9000개)을 기반으로 milvus 구성 및 vector search
    - Embedding Model : "snunlp/KR-SBERT-V40K-klueNLI-augSTS" (768 dim)
    - 블로그 포스팅 : 

2. milvus_cohere.ipynb
    - 1번에서 세팅한 기사 데이터셋(BalancedNewsCorpus)에 한국어 임베딩 성능이 좋다는 cohere의 모델을 적용
    - Embedding Model : "Cohere/embed-multilingual-v3.0" (1024 dim, search_type = ['search_query', 'search_document'])
    - 블로그 포스팅 :
  
3. milvus_upstage.ipynb
    - 1번에서 세팅한 기사 데이터셋(BalancedNewsCorpus)에 한국어 임베딩 성능이 좋다는 upstage의 모델을 적용
    - Embedding Model : "Upstage/solar-embedding-1-large-query", "Upstage/solar-embedding-1-large-passage" (4096 dim)
    - 블로그 포스팅 :
  
4. milvus_800.ipynb
    - 동아닷컴의 6월 1일부터 7월 17일까지의 기사(약 9000개)를 크롤링하여 구성한 DB를 바탕으로 upstage의 임베딩 모델을 적용
    - Embedding Model : "Upstage/solar-embedding-1-large-query", "Upstage/solar-embedding-1-large-passage" (4096 dim)
    - 블로그 포스팅 :

<br>

## Analysis
1. 서울대학교의 Sentence BERT, Cohere의 embed-multilingual-v3.0, Upstage의 solar-embedding-1-large를 비교해보았을 때, 벡터 검색 성능이 가장 좋다고 여기지는 것은 Upstage의 solar-embedding-1-large인 것 같다(Human Evaluation)

2. Cohere, Upstage의 최근 임베딩 모델들의 경우 DPR에서 착안한 것인지 query용, passage용 임베딩 모델이 구분된다
3. <a href="https://cohere.com/blog/introducing-embed-v3">Cohere Documentation</a>에 따르면, document용 임베딩 모델은 Document의 퀄리티도 반영하여 벡터값을 생성한다고 하는데 성능이 좋지 않아서 잘 모르겠다
4. Milvus에서 Index_Params에 들어가는 metric_type(['L2', 'IP', 'COSINE'])의 경우 위 세 가지 임베딩 모델에 대해서는 바꾸어도 검색 결과에 차이가 없다
    - IP, L2, COSINE의 경우 만약 사용하는 벡터가 정규화된(normalized) 벡터인 경우에는 IP와 COSINE은 일치하고 L2도 COSINE으로 표현된다 ( 관련 포스팅 )
    - 위 세 가지 임베딩 모델(서울대 SBERT, Cohere, Upstage) 모두 정규화된 벡터를 반환하는 것을 확인했다
    - 따라서 세 임베딩 모델을 사용하는 경우 Milvus의 metric_type을 바꿔가면서 검색 성능을 비교하는 것은 무의미한 일이다
  
```python
# 임베딩 벡터의 정규화 유무 확인

import numpy as np

embedding_vector = [0.003940582275390625,-0.00004583597183227539,-0.01416015625, ...] # 실제 임베딩된 값 중 하나에 테스트
norm = np.linalg.norm(embedding_vector)
print(norm) # 1에 근접하면 해당 임베딩 모델은 정규화된 임베딩 벡터를 반환
```
