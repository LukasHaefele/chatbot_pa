import dspy
import json

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
    api_key=""
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
    
    if bool(pred.answerable) != bool(example.answerable):
        return 0.0
    
    expected_src = set(example.expected_sources)
    pred_src = set(pred.sources or [])
    if not expected_src:
        source_score = 1.0 if not pred_src else 0.0
    else:
        source_score = len(expected_src & pred_src) / len(expected_src | pred_src)
    
    points = example.expected_answer_points
    graded = judge(
        question=example.question,
        predicted_answer=pred.answer,
        expected_points = points
    )
    point_score = min(max(int(graded.covered_points_count), 0), len(points)) / len(points)
    
    return 0.3 * source_score + 0.7 * point_score

models_list = [
    "openai/gpt-4o-mini",
    "ollama_chat/llama3.2"
    ]

lm = dspy.LM(
    models_list[1],
    api_base="http://localhost:11434",
    api_key=""
    )
dspy.configure(lm=lm)

chatbot = OnePlatformChatbot()

evaluator = dspy.Evaluate(devset=devset, metric=rag_metric, num_threads=4, display_progress=True)
baseline_score = evaluator(chatbot)

optimizer = dspy.MIPROv2(metric=rag_metric, auto="light")
opitmized_chatbot = optimizer.compile(chatbot, trainset=trainset, valset=devset)

test_score = evaluator(opitmized_chatbot, devset=testset)
opitmized_chatbot.save("optimized_chatbot.json")
