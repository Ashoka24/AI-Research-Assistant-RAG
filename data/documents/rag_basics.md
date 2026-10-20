# Retrieval-Augmented Generation

Retrieval-augmented generation combines a retrieval step with an answer-generation step. A system searches a selected document collection and uses relevant passages as context. A typical workflow loads documents, splits text into chunks, indexes those chunks, retrieves evidence for a question, and presents an answer with source references.

Chunk size is a trade-off: short chunks may lose context, while long chunks can include irrelevant text. TF-IDF is a useful lexical baseline for small collections, but it may miss semantic matches. Dense retrieval can be evaluated as an alternative rather than assumed to be better.

Retrieval does not guarantee correctness. The system can miss important passages, retrieve irrelevant text or present a summary that overstates what the source supports. Evaluate retrieval and answer quality separately, and verify citations against the original documents.
