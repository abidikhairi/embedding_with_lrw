import argparse
import numpy as np

from pymilvus import connections, FieldSchema, DataType, CollectionSchema, Collection

connections.connect(host='localhost', port='19530')

def main(args):
    collection_name = args.collection_name
    embedding_file = args.embedding_file

    embeddings = np.load(embedding_file)
    idx = np.arange(len(embeddings))
    
    embed_dim = embeddings.shape[1]
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embed_dim)
    ]
    
    schema = CollectionSchema(fields=fields, description="nodes embeding collection")
    
    collection = Collection(name=collection_name, schema=schema)

    collection.insert([idx.tolist(), embeddings.tolist()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='store embeddings to milvus database')

    parser.add_argument('--collection_name', type=str, required=True, help='collection name')
    parser.add_argument('--embedding_file', type=str, required=True, help='embdding file')

    args = parser.parse_args()

    main(args)
