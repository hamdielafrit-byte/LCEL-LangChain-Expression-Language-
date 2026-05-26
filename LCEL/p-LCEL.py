import re
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

# Optimized 1-Billion parameter model for speed on local machines
OLLAMA_MODEL = "llama3.2:1b"

llm_decomposer = ChatOllama(model=OLLAMA_MODEL, temperature=0.0, num_predict=60)
llm_answerer = ChatOllama(model=OLLAMA_MODEL, temperature=0.1, num_predict=120)
llm_combiner = ChatOllama(model=OLLAMA_MODEL, temperature=0.0, num_predict=150)

# ==========================================
# 1) Decomposer Stage (Strictly Factual)
# ==========================================
decompose_prompt = PromptTemplate.from_template(
    "You are a technical assistant. Break down the user question into up to 3 short, distinct sub-questions. "
    "Output ONLY a numbered list (1., 2., 3.). Do not write an introduction or apology.\n\n"
    "Question:\n{question}"
)
decomposer = decompose_prompt | llm_decomposer

def parse_numbered_subquestions(base_msg):
    text = getattr(base_msg, "content", str(base_msg)).strip()
    lines = re.split(r"\r?\n", text)
    subqs = []
    for line in lines:
        m = re.match(r"\s*\d+\s*[.)]\s*(.*\S.*)$", line)
        if m:
            subqs.append(m.group(1).strip())
    if not subqs and text:
        subqs = [text]
    
    # Assignment Output helper
    print("\n=== [ASSIGNMENT STEP 1] Decomposed Sub-Questions ===")
    for i, q in enumerate(subqs[:3], start=1):
        print(f"{i}. {q}")
        
    return subqs[:3]

parse_subq_runnable = RunnableLambda(parse_numbered_subquestions)

# ==========================================
# 2) Sub-question Answering Stage (Template Enforcement)
# ==========================================
answer_prompt = PromptTemplate.from_template(
    "Provide a technical answer to the sub-question. Follow this format exactly:\n"
    "Answer: <one-line summary>\n"
    "Steps:\n"
    "- <step1>\n"
    "- <step2>\n\n"
    "Sub-question: {subq}"
)
answer_chain = answer_prompt | llm_answerer

def run_answers(subquestions):
    inputs = [{"subq": q} for q in subquestions]
    outputs = answer_chain.batch(inputs)  # Runs concurrently using batch
    
    parsed = []
    print("\n=== [ASSIGNMENT STEP 2] Sub-Answers Plain Text ===")
    for out in outputs:
        text = getattr(out, "content", str(out)).strip()
        answer_line = None
        steps = []
        for line in text.splitlines():
            if line.lower().startswith("answer:"):
                answer_line = line.split(":", 1)[1].strip()
            elif re.match(r"\s*[-•]\s+", line):
                steps.append(re.sub(r"^\s*[-•]\s+", "", line).strip())
        
        item = {
            "answer": answer_line or text, 
            "steps": steps or ["Optimize execution parameters."], 
            "raw": text
        }
        parsed.append(item)
        
        # Print block matching assignment template
        print(f"Answer: {item['answer']}")
        print("Steps:")
        for s in item['steps']:
            print(f"- {s}")
            
    return parsed

run_answers_runnable = RunnableLambda(run_answers)

# ==========================================
# 3) Combiner / Synthesis Stage (Format Lock)
# ==========================================
combine_prompt = PromptTemplate.from_template(
    "Synthesize the provided sub-answers into the exact three-line format below. "
    "Do not include introductory words, remarks, or metadata.\n\n"
    "Input Data:\n{subanswers_text}\n\n"
    "Required Output Format Structure:\n"
    "1) Final Answer: <one line summary answer>\n"
    "2) Key points: - <point 1>; - <point 2>\n"
    "3) Confidence: high"
)

def format_subanswers_block(ans_list):
    blocks = []
    for i, a in enumerate(ans_list, start=1):
        blocks.append(f"{i}. Answer: {a['answer']}")
        blocks.append("   Steps:")
        for s in a["steps"]:
            blocks.append(f"   - {s}")
    return "\n".join(blocks)

format_runnable = RunnableLambda(lambda answers: {"subanswers_text": format_subanswers_block(answers)})
combiner = format_runnable | combine_prompt | llm_combiner

# ==========================================
# Pipeline Composition (LCEL)
# ==========================================
pipeline = decomposer | parse_subq_runnable | run_answers_runnable | combiner

if __name__ == "__main__":
    q = "How can I reduce latency in a web app that serves ML predictions?"
    print(f"Processing via structured LCEL pipeline using {OLLAMA_MODEL}...")
    
    start_time = time.time()
    final = pipeline.invoke({"question": q})
    
    print("\n=== [ASSIGNMENT STEP 3] Final 3-Line Synthesis ===")
    print(getattr(final, "content", str(final)).strip())
    print(f"\n⚡ Total Execution Time: {time.time() - start_time:.2f} seconds")