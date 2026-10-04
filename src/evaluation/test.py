"""
RAG Doctor evaluation test cases.

These are NOT "exact answer" benchmarks.
Each case describes a realistic RAG engineering problem and defines:
- concepts the answer should recognize
- strategies that would be reasonable to discuss
- an example of the ideal answer structure/content
- optional ground-truth chunk IDs for classical Recall@K / MRR

Start with 10-15 cases. Add real chunk IDs only after inspecting your corpus.
"""

TEST_CASES = [
    {
        "id": "TC001",
        "category": "retrieval",
        "question": (
            "My retriever usually returns relevant chunks, but the most useful "
            "chunk is often ranked below several less useful chunks. What should "
            "I change to improve the ranking?"
        ),
        "expected_concepts": [
            "retrieval relevance",
            "ranking",
            "reranking",
            "candidate retrieval",
        ],
        "expected_strategies": [
            "reranking retrieved candidates",
            "retrieve a larger candidate set before reranking",
            "compare ranking quality with nDCG or MRR",
        ],
        "ideal_answer": (
            "Diagnose this as a ranking problem. Explain that dense retrieval "
            "can produce a useful candidate set without putting the best evidence "
            "first. Discuss reranking and a larger candidate pool, then propose "
            "an experiment comparing the original retriever with reranking."
        ),
        "relevant_chunk_ids": [],
    },
    {
        "id": "TC002",
        "category": "chunking",
        "question": (
            "My chunks are very large. Retrieval finds the right document, but "
            "each retrieved chunk contains a lot of unrelated information and "
            "the final answer becomes noisy. How should I improve this?"
        ),
        "expected_concepts": [
            "chunk size",
            "retrieval precision",
            "context noise",
            "semantic boundaries",
        ],
        "expected_strategies": [
            "smaller or more focused chunks",
            "semantic or structure-aware chunking",
            "evaluate different chunk sizes",
        ],
        "ideal_answer": (
            "Explain that overly large chunks can reduce retrieval precision and "
            "introduce irrelevant context. Recommend testing smaller, semantically "
            "coherent chunks while preserving enough context, and compare retrieval "
            "and answer metrics across chunking configurations."
        ),
        "relevant_chunk_ids": [],
    },
    {
        "id": "TC003",
        "category": "chunking",
        "question": (
            "Important information is sometimes split across two neighboring "
            "chunks, so retrieving only one chunk gives an incomplete answer. "
            "What can I do?"
        ),
        "expected_concepts": [
            "chunk boundaries",
            "context continuity",
            "overlap",
            "parent-child or neighboring context",
        ],
        "expected_strategies": [
            "chunk overlap",
            "retrieve neighboring chunks",
            "parent-child or hierarchical retrieval",
        ],
        "ideal_answer": (
            "Identify boundary fragmentation as the likely issue. Discuss overlap "
            "or retrieving adjacent/parent context, and propose measuring answer "
            "completeness together with retrieval metrics."
        ),
        "relevant_chunk_ids": [],
    },
    {
        "id": "TC004",
        "category": "retrieval",
        "question": (
            "My users describe a concept using different words from the research "
            "papers, so dense retrieval sometimes misses the relevant evidence. "
            "How can I make retrieval more robust to terminology differences?"
        ),
        "expected_concepts": [
            "query-document vocabulary mismatch",
            "semantic retrieval",
            "lexical retrieval",
            "query expansion",
            "hybrid retrieval",
        ],
        "expected_strategies": [
            "hybrid dense and lexical retrieval",
            "query expansion or rewriting",
            "rerank combined candidates",
        ],
        "ideal_answer": (
            "Explain vocabulary mismatch and compare semantic retrieval with "
            "lexical signals. Recommend hybrid retrieval and/or query rewriting, "
            "followed by reranking when appropriate."
        ),
        "relevant_chunk_ids": [],
    },
    {
        "id": "TC005",
        "category": "retrieval",
        "question": (
            "Increasing top-k from 5 to 15 increases the chance of retrieving "
            "useful evidence, but the final answer gets worse because the LLM "
            "receives too much irrelevant context. How should I handle this trade-off?"
        ),
        "expected_concepts": [
            "top-k",
            "context noise",
            "precision-recall trade-off",
            "reranking",
            "context selection",
        ],
        "expected_strategies": [
            "retrieve more candidates and rerank",
            "tune final context size separately from candidate k",
            "evaluate answer quality at different k values",
        ],
        "ideal_answer": (
            "Separate candidate retrieval from final context selection. Explain "
            "the precision-recall trade-off and recommend retrieving a broader "
            "candidate set, reranking it, and sending only the strongest evidence "
            "to the generator."
        ),
        "relevant_chunk_ids": [],
    },
    {
        "id": "TC006",
        "category": "generation",
        "question": (
            "My RAG retrieves relevant chunks, but the generated answer still "
            "contains claims that are not supported by the retrieved context. "
            "How can I improve grounding?"
        ),
        "expected_concepts": [
            "faithfulness",
            "grounding",
            "hallucination",
            "context-supported generation",
        ],
        "expected_strategies": [
            "constrain generation to retrieved evidence",
            "improve evidence selection",
            "answer verification or claim checking",
        ],
        "ideal_answer": (
            "Distinguish retrieval quality from generation faithfulness. Explain "
            "that relevant retrieval does not guarantee a grounded answer. "
            "Recommend evidence-focused prompting, better context selection, "
            "and answer/claim verification."
        ),
        "relevant_chunk_ids": [],
    },
    {
        "id": "TC007",
        "category": "retrieval",
        "question": (
            "My retrieval metrics look good, but users still say the answers "
            "are incomplete. What should I investigate next?"
        ),
        "expected_concepts": [
            "retrieval quality versus generation quality",
            "context completeness",
            "answer completeness",
            "faithfulness",
        ],
        "expected_strategies": [
            "evaluate answer completeness separately",
            "inspect whether multiple evidence pieces are needed",
            "evaluate generation and retrieval independently",
        ],
        "ideal_answer": (
            "Explain that good retrieval metrics do not automatically imply good "
            "answers. Inspect whether the retrieved context contains all necessary "
            "evidence, then separately evaluate completeness, faithfulness, and "
            "answer relevance."
        ),
        "relevant_chunk_ids": [],
    },
    {
        "id": "TC008",
        "category": "documents",
        "question": (
            "My RAG works well on normal paragraphs but performs poorly when the "
            "answer is contained inside a table in a research paper. What should "
            "I change?"
        ),
        "expected_concepts": [
            "table extraction",
            "structured document content",
            "table-aware retrieval",
            "document parsing",
        ],
        "expected_strategies": [
            "preserve table structure during ingestion",
            "index table content appropriately",
            "use table-aware retrieval or representation",
        ],
        "ideal_answer": (
            "Treat tables as structured evidence rather than ordinary paragraphs. "
            "Check whether ingestion preserved the table correctly, then evaluate "
            "how table content is represented and retrieved."
        ),
        "relevant_chunk_ids": [],
    },
    {
        "id": "TC009",
        "category": "query",
        "question": (
            "Users ask vague questions such as 'How can I improve my RAG?', and "
            "retrieval becomes inconsistent. What should happen before retrieval?"
        ),
        "expected_concepts": [
            "query understanding",
            "query rewriting",
            "query decomposition",
            "intent detection",
        ],
        "expected_strategies": [
            "query rewriting",
            "intent-aware query processing",
            "query decomposition for multi-part problems",
        ],
        "ideal_answer": (
            "Diagnose ambiguity in the input query. Recommend query rewriting or "
            "decomposition and, where useful, intent-aware processing before retrieval."
        ),
        "relevant_chunk_ids": [],
    },
    {
        "id": "TC010",
        "category": "evaluation",
        "question": (
            "I changed my chunking strategy and want to know whether retrieval "
            "actually improved. Which evaluation approach should I use?"
        ),
        "expected_concepts": [
            "retrieval evaluation",
            "Recall@K",
            "Precision@K",
            "MRR",
            "nDCG",
        ],
        "expected_strategies": [
            "compare retrieval metrics before and after",
            "use graded relevance when chunks have different usefulness",
            "evaluate answer quality separately",
        ],
        "ideal_answer": (
            "Recommend evaluating retrieval independently from generation. "
            "Use Recall@K and Precision@K for coverage/precision, MRR when the "
            "first relevant result matters, and nDCG when relevance is graded."
        ),
        "relevant_chunk_ids": [],
    },
]


if __name__ == "__main__":
    for case in TEST_CASES:
        print(f"\n{case['id']} — {case['category']}")
        print(case["question"])
        print("Expected concepts:", ", ".join(case["expected_concepts"]))
        print("Expected strategies:", ", ".join(case["expected_strategies"]))