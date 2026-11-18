import dspy
import json
from datetime import datetime
from typing import Literal
from lingua import Language, LanguageDetectorBuilder

language_detector = (
    LanguageDetectorBuilder
    .from_languages(Language.GERMAN, Language.ENGLISH)
    .with_preloaded_language_models()
    .build()
)

def detect_language(text: str) -> str:
    """Gibt 'de' oder 'en' zurück, basierend auf dem Text (lokal, ohne LLM)."""
    if not text or not text.strip():
        return "unknown"
    lang = language_detector.detect_language_of(text)
    if lang == Language.GERMAN:
        return "de"
    if lang == Language.ENGLISH:
        return "en"
    return "unknown"
    
with open("one-platform-chatbot-dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

documents = raw_data["documents"]

splits = {"train": [], "dev": [], "test": []}
for case in raw_data["cases"]:
    example = dspy.Example(
        id = case["id"],
        question = case["question"],
        language = case["language"],
        answerable = case["answerable"],
        expected_sources = case["expected_sources"],
        expected_answer_points = case["expected_answer_points"]
    ).with_inputs("question")

    splits[case["suggested_split"]].append(example)
    
trainset, devset, testset = splits["train"], splits["dev"], splits["test"]

corpus = [
    f"[{doc['id']}] {doc['title']}:\n{doc['content']}" for doc in documents
]

embedder = dspy.Embedder(
    "ollama/nomic-embed-text",
    api_base="http://localhost:11434",
    api_key="ollama"
    )
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=3)

class OnePlatformRAGSignature(dspy.Signature):
    """Answor questions about One Platform based ONLY on the provided context. Respond in the same language as the question (German or English). If the context does not contain the answer or asks for secrets/live status mark answerable as False
    """
    
    context: list[str] = dspy.InputField(desc="Retrieved knowledge base documents with [doc_id]")
    question: str = dspy.InputField(desc="User question in German or English")
    answerable: bool = dspy.OutputField(desc="True if answerable from context, False otherwise")
    sources: list[str] = dspy.OutputField(desc="List of document IDs used, e.g. ['kb-ews-cli'], or [] if unanswerable")
    answer: str = dspy.OutputField(desc="Concise answer covering the key points in the question's language")
    
class OnePlatformChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought(OnePlatformRAGSignature)
    
    def forward(self, question: str):
        context = search(question).passages
        return self.respond(context=context, question=question)
    
class GradeAnswerPoints(dspy.Signature):
    """Check which expected answer points are covered by the predicted answer."""
    question: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    expected_points: list[str] = dspy.InputField()
    covered_points_count: int = dspy.OutputField(desc="Number of expected_points accurately present in predicted_answer")
    
judge = dspy.Predict(GradeAnswerPoints)

def rag_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    
    pred.answerable_score = 1.0
    pred.source_score = 0.0
    pred.point_score = 0.0
    pred.correct_count = 0.0
    pred.incorrect_count = 0.0
    pred.missing_count = 0.0
    pred.language_score = 0.0
    if bool(pred.answerable) != bool(example.answerable):
        pred.answerable_score = 0.0
        # return 0.0
    
    
    expected_src = set(example.expected_sources)
    pred_src = set(pred.sources or [])
    correct_count = 0.0
    incorrect_count = 0.0
    for src in pred_src:
        if expected_src.__contains__(src):
            correct_count += 1.0
            expected_src.remove(src)
        else:
            incorrect_count += 1.0
    missing_count = len(expected_src)
    
    source_score = correct_count / max((missing_count + incorrect_count + correct_count), 1)
    pred.correct_count = correct_count
    pred.incorrect_count = incorrect_count
    pred.missing_count = missing_count
    pred.source_score = source_score
    
    points = example.expected_answer_points
    graded = judge(
        question=example.question,
        predicted_answer=pred.answer,
        expected_points = points
    )
    point_score = min(max(int(graded.covered_points_count), 0), len(points)) / len(points)
    pred.point_score = point_score

    detected_language = detect_language(pred.answer)
    language_score = 1.0 if detected_language == example.language else 0.0
    pred.answer_language = detected_language
    pred.language_score = language_score
    
    return 0.25 * source_score + 0.65 * point_score + 0.1 * language_score

prompt_lm = dspy.LM(
    "ollama_chat/gemma3:12b",
    api_base="http://localhost:11434",
    api_key="ollama",
    max_tokens=2000,
    num_ctx=8192,
    cache=False,
)

lm = dspy.LM(
    "ollama_chat/gemma3:12b",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
    cache=False
    )
dspy.configure(lm=lm)

chatbot = OnePlatformChatbot()

dtstring = datetime.now().strftime("%y-%m-%d_%H-%M")
eval_string = f"output/evaluator/dev/{dtstring}.json"

evaluator = dspy.Evaluate(devset=devset, metric=rag_metric, num_threads=4, display_progress=True, save_as_json=eval_string)
baseline_score = evaluator(chatbot)

auto = ["light","medium", "heavy"]
auto_index = 0
optimizer = dspy.MIPROv2(metric=rag_metric, auto=auto[auto_index], prompt_model=prompt_lm, task_model=lm)
opitmized_chatbot = optimizer.compile(chatbot, trainset=trainset, valset=devset)

test_string = f"output/evaluator/test/{dtstring}.json"
test_score = evaluator(opitmized_chatbot, devset=testset, num_threads=4, display_progress=True, save_as_json=test_string)
opitmized_chatbot.save(f"output/configs/optimized_chatbot_{auto[auto_index]}_{dtstring}.json")
