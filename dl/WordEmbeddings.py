from annoy import AnnoyIndex

# To query for nearest neighbors
index = AnnoyIndex(3, metric='euclidian')

# Show the surrounding words
