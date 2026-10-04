"""
RAG Doctor evaluation runner.

Pipeline:
    test case
       ↓
    BGE-M3 query embedding
       ↓
    Chroma retrieval
       ↓
    LLM relevance judge
       ↓
    Precision@K / Recall@K / MRR / nDCG
       ↓
    Gemini answer generation
       ↓
    Gemini answer judge
       ↓
    evaluation_results.json

Important:
- nDCG, Precision and MRR can be calculated from graded relevance judgments.
- True Recall@K requires a known relevant set. Add `relevant_chunk_ids` to a
  test case when you have manually verified ground-truth chunks.
- If `relevant_chunk_ids` is empty, the script reports Recall@K as None rather
  than pretending that an LLM's retrieved-set judgment is classical recall.
"""

from __future__ import annotations

import json
import math,traceback
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

import chromadb
from dotenv import load_dotenv
from litellm import completion, embedding
from sentence_transformers import SentenceTransformer

from src.evaluation.test import TEST_CASES


load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "vectorstore" / "chroma"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"

COLLECTION_NAME = os.getenv("RAG_DOCTOR_COLLECTION", "rag_doctor")
EMBEDDING_MODEL = os.getenv("RAG_DOCTOR_EMBEDDING_MODEL", "openrouter/openai/text-embedding-3-small")
LLM_MODEL = os.getenv("RAG_DOCTOR_MODEL", "openrouter/openai/gpt-oss-120b")
RERANKER_MODEL = os.getenv("RAG_DOCTOR_RERANKER_MODEL","openrouter/deepseek/deepseek-v4.1-flash")
TOP_K = int(os.getenv("RAG_DOCTOR_TOP_K", "10"))

load_dotenv(override=True)
load_dotenv(override=True)

print("EMBEDDING_MODEL =", os.getenv("RAG_DOCTOR_EMBEDDING_MODEL"))

# _embedding_model = None
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
class AnswerEvaluation(BaseModel):
    problem_understanding: int = Field(ge=1, le=5, description="How well the answer understands and addresses the user's actual problem")
    faithfulness: int = Field(ge=1,le=5,description="How well the claims in the answer are supported by the retrieved context")
    relevance: int = Field(ge=1,le=5,description="How directly the answer addresses the user's problem")
    completeness: int = Field(ge=1,le=5,description="How well the answer covers the important concepts required to solve the problem")
    recommendation_quality: int = Field(ge=1,le=5,description="How appropriate and actionable the recommended strategies are")
    experiment_quality: int = Field(ge=1,le=5,description="How useful and appropriate the proposed validation or experiment is")
    citation_support: int = Field(ge=1,le=5,description="How well evidence-based claims can be traced back to the retrieved context")
    overall_comment: str = Field(description="Short explanation summarizing the strengths and weaknesses of the answer")

class RankOrder(BaseModel):
    order: list[str] = Field(description="Chunk IDs ordered from most relevant to least relevant")    

# def get_embedding_model() -> SentenceTransformer:
#     global _embedding_model
#     if _embedding_model is None:
#         _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
#     return _embedding_model


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(COLLECTION_NAME)


def retrieve(question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    collection = get_collection()

    response = embedding(
        model=EMBEDDING_MODEL,
        input=[question],
    )

    query_vector = response.data[0]["embedding"]

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result.get("distances", [[]])[0]
    ids = result["ids"][0]

    chunks = []

    for rank, (chunk_id, document, metadata) in enumerate(
        zip(ids, documents, metadatas),
        start=1,
    ):
        chunks.append(
            {
                "rank": rank,
                "id": chunk_id,
                "document": document or "",
                "metadata": metadata or {},
                "distance": distances[rank - 1] if distances else None,
            }
        )

    return chunks

def rerank(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_prompt = """
    You are a document re-ranker.
    
    You are provided with a question and a list of relevant chunks of text
    retrieved from a knowledge base.
    
    The chunks are provided in their original retrieval order. This order is
    approximately based on relevance, but you may improve it.
    
    Your task is ONLY to reorder the provided chunks by relevance to the question.
    Return the chunk IDs from most relevant to least relevant.
    Include every chunk ID exactly once.
    Do not add, remove, or modify any chunk.
    Reply only with the JSON structure requested.
    """

    user_prompt = f"""
    The user has asked the following question:
    
    {question}
    
    Order all the chunks by relevance to the question,
    from most relevant to least relevant.
    
    Include every chunk ID exactly once.
    
    Here are the chunks:
    """

    for chunk in chunks:
        user_prompt += (
            f"# CHUNK ID: {chunk['id']}\n\n"
            f"{chunk['document']}\n\n"
        )

    raw = _call_llm(
        system_prompt,
        user_prompt,
        model=RERANKER_MODEL,
        response_format=RankOrder
    )

    evaluation = RankOrder.model_validate_json(raw)

    expected_ids = {chunk["id"] for chunk in chunks}

    if (
        set(evaluation.order) != expected_ids
        or len(evaluation.order) != len(expected_ids)
    ):
        raise ValueError(
            "Reranker did not return every chunk ID exactly once."
        )

    chunks_by_id = {
        chunk["id"]: chunk
        for chunk in chunks
    }

    reranked = [
        chunks_by_id[chunk_id]
        for chunk_id in evaluation.order
        if chunk_id in chunks_by_id
    ]

    return [
        {**chunk, "rank": rank}
        for rank, chunk in enumerate(reranked, start=1)
        ]



def _call_llm(system_prompt: str, user_prompt: str, model: str = LLM_MODEL, response_format: type[BaseModel] | None = None) -> str:
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format=response_format,
    )

    return response.choices[0].message.content

class RetrievalJudger(BaseModel):
    relevance: int = Field(
        ge=0,
        le=3,
        description="0=irrelevant, 1=weakly relevant, 2=relevant, 3=highly relevant"
    )
    reason: str = Field(
        description="Short reason for the relevance score"
    )


class RetrievalEvaluation(BaseModel):
    judgments: list[RetrievalJudger] = Field(
        description="Judgment for every chunk, in exactly the same order as provided"
    )


def judge_retrieval(case: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunk_text = "\n\n".join(
        f"CHUNK {chunk['rank']} | id={chunk['id']}\n{chunk['document'][:2500]}"
        for chunk in chunks
    )

    prompt = f"""
        Evaluate how useful each retrieved chunk is for solving the user's RAG
        engineering problem.

        USER PROBLEM:

        {case["question"]}

        EXPECTED CONCEPTS:

        {json.dumps(case["expected_concepts"])}

        EXPECTED STRATEGIES:

        {json.dumps(case["expected_strategies"])}

        RETRIEVED CHUNKS:

        {chunk_text}

        Score every chunk:

        0 = irrelevant

        1 = weakly relevant / background only

        2 = relevant evidence

        3 = highly relevant evidence directly useful for diagnosing or solving the problem.

        For every chunk, return:
        - rank
        - chunk_id
        - relevance
        - short reason

        Evaluate every retrieved chunk.
        """
    system = (
        "You are a strict retrieval evaluator. "
        "Judge only the usefulness of each chunk for the stated problem. "
        "Do not reward a chunk merely because it discusses RAG generally."
    )

    raw = _call_llm(system, prompt, response_format=RetrievalEvaluation, model=RERANKER_MODEL)
    evaluation = RetrievalEvaluation.model_validate_json(raw)

    return [
        {
            **chunk,
            "relevance": judgment.relevance,
            "relevance_reason": judgment.reason,
        }
        for chunk, judgment in zip(chunks, evaluation.judgments)
    ]


def precision_at_k(relevances: list[int], k: int, threshold: int = 2) -> float:
    values = relevances[:k]
    if not values:
        return 0.0
    return sum(score >= threshold for score in values) / len(values)


def dcg(relevances: list[int]) -> float:
    return sum(
        (2**rel - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances, start=1)
    )


def ndcg_at_k(relevances: list[int], k: int) -> float:
    actual = dcg(relevances[:k])
    ideal = dcg(sorted(relevances, reverse=True)[:k])

    if ideal == 0:
        return 0.0

    return actual / ideal


def mrr(relevances: list[int], threshold: int = 2) -> float:
    for rank, relevance in enumerate(relevances, start=1):
        if relevance >= threshold:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float | None:
    if not relevant_ids:
        return None

    relevant_set = set(relevant_ids)
    retrieved = set(retrieved_ids[:k])

    return len(retrieved & relevant_set) / len(relevant_set)


def retrieval_metrics(
    judged_chunks: list[dict[str, Any]],
    relevant_ids: list[str],
) -> dict[str, Any]:
    relevances = [chunk["relevance"] for chunk in judged_chunks]
    ids = [chunk["id"] for chunk in judged_chunks]

    return {
        "precision_at_5": precision_at_k(relevances, 5),
        "precision_at_10": precision_at_k(relevances, 10),
        "recall_at_5": recall_at_k(ids, relevant_ids, 5),
        "recall_at_10": recall_at_k(ids, relevant_ids, 10),
        "mrr": mrr(relevances),
        "ndcg_at_5": ndcg_at_k(relevances, 5),
        "ndcg_at_10": ndcg_at_k(relevances, 10),
    }


def generate_answer(
    case: dict[str, Any],
    judged_chunks: list[dict[str, Any]],
) -> str:
    context = "\n\n".join(
        (
            f"[{chunk['rank']}] "
            f"{chunk['metadata'].get('doc_id', 'unknown')} | "
            f"{chunk['metadata'].get('section_path', 'unknown')}\n"
            f"{chunk['document']}"
        )
        for chunk in judged_chunks[:5]
    )

    prompt = f"""
You are RAG Doctor, an engineering advisor for Retrieval-Augmented Generation.

The user may:
- describe a RAG problem,
- ask for a recommendation,
- ask how to improve their RAG,
- compare RAG techniques,
- or ask how to implement a RAG strategy.

First determine the user's intent.

Possible intents:
1. troubleshooting
2. recommendation
3. optimization
4. comparison
5. implementation
6. conceptual

Adapt the response structure to the intent.

For troubleshooting:
- Diagnosis
- Why It Happens
- Recommended Strategies
- Implementation
- Evidence
- Sources

For recommendation:
- Recommended Strategy
- Why
- Alternatives
- Suggested Experiment
- Evidence
- Sources

For comparison:
- Option A
- Option B
- Trade-offs
- Recommendation for the stated use case
- Evidence
- Sources

Do not force irrelevant sections into the response.

Base recommendations on the retrieved research context.
If the research does not provide enough evidence for a specific
recommendation, clearly distinguish established evidence from
your own general engineering suggestion.

USER PROBLEM:
{case["question"]}

RETRIEVED RESEARCH CONTEXT:
{context}
"""

    return _call_llm(
        "You are a careful, evidence-grounded RAG engineering advisor.",
        prompt,
    )


def judge_answer(
    case: dict[str, Any],
    answer: str,
    judged_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    context = "\n\n".join(
        f"CHUNK {chunk['rank']}:\n{chunk['document'][:2500]}"
        for chunk in judged_chunks[:5]
    )

    prompt = f"""
Score each dimension from 1 to 5:
- problem_understanding
- faithfulness: claims are supported by retrieved context
- relevance: directly addresses the user's problem
- completeness: covers important concepts
- recommendation_quality: strategies are appropriate and actionable
- experiment_quality: proposed validation/experiment is useful
- citation_support: evidence claims can be traced to retrieved context

Evaluate this RAG Doctor answer.

USER PROBLEM:
{case["question"]}

EXPECTED CONCEPTS:
{json.dumps(case["expected_concepts"])}

EXPECTED STRATEGIES:
{json.dumps(case["expected_strategies"])}

IDEAL ANSWER DIRECTION:
{case["ideal_answer"]}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

"""

    raw = _call_llm(system_prompt=(
            "You are a strict LLM-as-judge for a RAG engineering system. "
            "Do not reward fluent writing when evidence is missing."
        ), user_prompt=prompt, response_format=AnswerEvaluation,model="openrouter/google/gemini-2.5-flash-lite")
     
    return AnswerEvaluation.model_validate_json(raw)


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    print("  → Retrieving...")
    chunks = retrieve(case["question"])

    print("  → Reranking...")
    reranked_chunks = rerank(case["question"], chunks)

    print("  → Judging retrieval...")
    judged_chunks = judge_retrieval(case, reranked_chunks)

    print("  → Calculating metrics...")
    metrics = retrieval_metrics(
        judged_chunks,
        case.get("relevant_chunk_ids", []),
    )

    print("  → Generating answer...")
    answer = generate_answer(case, judged_chunks)

    print("  → Judging answer...")
    answer_judge = judge_answer(case, answer, judged_chunks)

    print("  ✓ TC complete")

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "retrieval_metrics": metrics,
        "answer": answer,
        "answer_judge": answer_judge.model_dump(),
        "retrieved_chunks": judged_chunks,
    }


def run_evaluation(
    test_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    test_cases = test_cases or TEST_CASES

    results = []

    for index, case in enumerate(test_cases, start=1):
        print(f"[{index}/{len(test_cases)}] {case['id']}")

        try:
            results.append(run_case(case))
        except Exception as exc:
            traceback.print_exc()
            
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "error": str(exc),
                }
            )
    successful = [item for item in results if "error" not in item]

    summary = {
        "num_test_cases": len(results),
        "successful_cases": len(successful),
        "retrieval": {
            "mean_precision_at_5": _mean(
                item["retrieval_metrics"]["precision_at_5"]
                for item in successful
            ),
            "mean_ndcg_at_5": _mean(
                item["retrieval_metrics"]["ndcg_at_5"]
                for item in successful
            ),
            "mean_mrr": _mean(
                item["retrieval_metrics"]["mrr"]
                for item in successful
            ),
            "mean_recall_at_10": _mean_optional(
                item["retrieval_metrics"]["recall_at_10"]
                for item in successful
            ),
        },
        "answer": {
            key: _mean(
                item["answer_judge"][key]
                for item in successful
                if key in item["answer_judge"]
            )
            for key in [
                "problem_understanding",
                "faithfulness",
                "relevance",
                "completeness",
                "recommendation_quality",
                "experiment_quality",
                "citation_support",
            ]
        },
    }

    output = {
        "summary": summary,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nSaved evaluation results to: {RESULTS_FILE}")

    return output


def _mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def _mean_optional(values) -> float | None:
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


if __name__ == "__main__":
    run_evaluation()
