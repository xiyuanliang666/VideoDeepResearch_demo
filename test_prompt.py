import os
import openai
import csv
import json
import re
from tavily import TavilyClient
openai.api_key = "sk-n3qDgJr5AjZ9bBCdC727278121Ba4066Aa5c1d329dB1Ed85"
openai.base_url = "https://api.gpt.ge/v1/"
openai.default_headers = {"x-foo": "true"}
def init_vision_model():
    import accelerate  # 使用 device_map="auto" 必须先导入 accelerate
    from transformers import AutoProcessor, Glm4vForConditionalGeneration
    import torch
    import time

    MODEL_PATH = "/mnt/sda/wangyu/models--zai-org--GLM-4.6V-Flash/snapshots/411bb4d77144a3f03accbf4b780f5acb8b7cde4e"
    
    print("=" * 50)
    print("Loading vision model...")
    print("=" * 50)
    
    start_time = time.time()
    print(f"[1/2] Loading processor from {MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"      Processor loaded in {time.time() - start_time:.2f}s")
    
    model_start = time.time()
    print(f"[2/2] Loading model (this may take a few minutes)...")
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path="zai-org/GLM-4.6V-Flash",
        torch_dtype=torch.bfloat16,  # 明确指定类型，比 "auto" 更快
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"      Model loaded in {time.time() - model_start:.2f}s")
    print(f"      Total loading time: {time.time() - start_time:.2f}s")
    print("=" * 50)
    print("Model ready!")
    print("=" * 50)
    
    return processor, model
# 模拟的视频观看功能
def observe_video(videopath="",question="", processor=None, model=None):
    """
    模拟视频观看功能
    在实际应用中，这里应该调用真实的视频分析API
    """
    # print(f"\n[Executing observe_video]")
    # print(f"  Time range: {time_range if time_range else 'Not specified'}")
    # print(f"  Observation content: {description}")
    
    # # Simulate return: Generate reasonable observation results based on description
    # # In actual application, this should return real video analysis results
    # result = f"Based on video observation, {description}. Relevant content has been observed."
    
    # # If asking about tool name, simulate returning a tool
    # if "tool" in description.lower():
    #     result = "Observed from the video that this is a traditional handcraft tool making process. Based on the crafting techniques and tool characteristics in the video, this appears to be a Japanese traditional tool."
    
    # print(f"  Observation result: {result}")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": videopath
                },
                {
                    "type": "text",
                    "text": question
                }
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    return output_text

# 模拟的网络搜索功能
def search_web(query):
    # """
    # 模拟网络搜索功能
    # 在实际应用中，这里应该调用真实的搜索引擎API
    # """
    # print(f"\n[Executing search_web]")
    # print(f"  Search query: {query}")
    
    # # Simulate search results
    # # In actual application, this should return real search results
    # result = f"Based on web search '{query}', found relevant information."
    
    # # Simulate return results for specific queries
    # if "meiji" in query.lower():
    #     result = "Search result: This tool was invented during the Meiji era. According to official records, the invention year was Meiji XX."
    # elif "invented" in query.lower():
    #     result = "Search result: Found historical information about the tool's invention."
    
    # print(f"  Search result: {result}")
    search_client = TavilyClient(api_key="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")
    response = search_client.search(query)
    results = []
        
    for item in response["results"]:
        results.append(
            {
            "url": item["url"],
            "title": item["title"],
            "content": item["content"],
            "score": item["score"]
        }
    )
    
    # 将结果转换为纯文本格式
    text_result = "Web search results:\n\n"
    for idx, item in enumerate(results, 1):
        text_result += f"Result {idx} (Score: {item['score']:.4f}):\n"
        text_result += f"Title: {item['title']}\n"
        text_result += f"URL: {item['url']}\n"
        text_result += f"Content: {item['content']}\n\n"
    
    return text_result.strip()

# 模拟的证据阅读功能
def read_evidence(source_id):
    """
    模拟证据阅读功能
    在实际应用中，这里应该读取真实的搜索结果
    """
    print(f"\n[Executing read_evidence]")
    print(f"  Evidence source: {source_id}")
    
    result = f"Extracted relevant information from evidence source {source_id}."
    print(f"  Evidence content: {result}")
    return result

# 推理功能
def reason(input_text):
    """
    推理功能，使用模型进行推理
    """
    print(f"\n[Executing reason]")
    print(f"  Reasoning content: {input_text}")
    
    # Use model for reasoning
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a reasoning assistant that performs logical reasoning based on provided information."
            },
            {
                "role": "user",
                "content": f"Please reason based on the following information: {input_text}"
            }
        ],
    )
    
    result = completion.choices[0].message.content
    print(f"  Reasoning result: {result}")
    return result

def get_system_prompt():
    """Get system prompt"""
    return """You are an intelligent video understanding assistant responsible for solving complex video-related questions step by step.

    CRITICAL: TASK DECOMPOSITION IS ESSENTIAL
    Before calling any tool, you MUST first decompose complex questions into smaller, focused sub-tasks. Never pass a complex multi-part question directly to a tool.

    STRICT CONSTRAINTS:
    1. You can ONLY call observe_video ONCE throughout the entire problem-solving process
    2. You MUST call search_web AT LEAST ONCE before providing the final answer

    Your workflow:
    1. ANALYZE and DECOMPOSE: Break down the complex question into smaller, specific sub-questions
    - Identify all distinct pieces of information needed
    - List each sub-question separately
    - Determine the order in which to answer them
    2. EXECUTE ONE SUB-TASK AT A TIME: Each tool call should address only ONE specific, focused question
    3. ACCUMULATE INFORMATION: Collect answers from each sub-task
    4. SYNTHESIZE: Once all sub-tasks are complete, combine the information to answer the original question
    5. ANSWER: Provide the final answer when all required information is gathered

    Task Decomposition Guidelines:
    - If a question asks about multiple items (e.g., "three buildings"), you should combine all visual observation needs into ONE observe_video call
    - If a question requires both video observation AND web search, do observe_video FIRST (only once), then use search_web
    - If a question asks for a comparison or calculation, first gather all individual facts, then perform the comparison/calculation
    - Each sub-question should be simple, specific, and answerable with a single tool call

    Available tools:
    - observe_video: Watch the video and answer questions about what you see. CAN ONLY BE CALLED ONCE.
    Input format: {"question": "A question about the video content"}
    IMPORTANT: You can only call this tool ONCE. Combine all visual observation needs into a single comprehensive question if needed.
    - search_web: Search for specific information on the web. MUST BE CALLED AT LEAST ONCE.
    Input format: A focused search query string (not a complex question)
    IMPORTANT: You MUST use this tool at least once before providing the final answer. Use specific, targeted search queries.
    - answer: When you have sufficient information, use this tool to provide the final answer
    Input format: Final answer string

    Important constraints:
    - ALWAYS decompose complex questions before calling tools
    - observe_video can ONLY be called ONCE - combine all visual observation needs if necessary
    - search_web MUST be called AT LEAST ONCE before providing the final answer
    - Execute only ONE focused sub-task per tool call
    - You must output your decision in JSON format as follows:
    {"action": "tool_name", "input": input_parameter, "reasoning": "reason for choosing this action", "sub_task": "description of this specific sub-task"}
    - Only use the answer tool when you have ALL required information to answer the original question AND you have called search_web at least once
    - For observe_video, input must be a dictionary with a question (can be comprehensive if combining multiple observations)
    - For search_web, input should be a specific, targeted search query (not a complex question)

    Example of proper task decomposition:
    Original question: "The video shows three buildings: (1) a castle; (2) a monument; (3) a pavilion. In which century was the earliest-built constructed?"

    Step 1 (Decomposition): Break into sub-tasks:
    - Sub-task 1: Observe the video ONCE to identify all three buildings and their features
    - Sub-task 2: Search the web for construction dates of each identified building (REQUIRED)
    - Sub-task 3: Compare dates and determine the earliest century

    Step 2 (Execution):
    {"action": "observe_video", "input": {"question": "What buildings appear in this video? Identify and describe: (1) What castle appears? What are its key features? (2) What monument with a high spire appears? What are its features? (3) What circular or octagonal viewing pavilion appears? What are its features?"}, "reasoning": "Need to identify all three buildings in one observation call", "sub_task": "Identify all buildings"}
    {"action": "search_web", "input": "[castle name from observation] construction date century", "reasoning": "Required: Need construction date for castle", "sub_task": "Find castle construction date"}
    {"action": "search_web", "input": "[monument name from observation] construction date century", "reasoning": "Required: Need construction date for monument", "sub_task": "Find monument construction date"}
    {"action": "search_web", "input": "[pavilion name from observation] construction date century", "reasoning": "Required: Need construction date for pavilion", "sub_task": "Find pavilion construction date"}
    {"action": "answer", "input": "11th century", "reasoning": "All information gathered and compared, search_web has been called", "sub_task": "Final answer"}

    Output format examples:
    {"action": "observe_video", "input": {"question": "What tool is being made in this video? What is the tool and what are its characteristics?"}, "reasoning": "Single observation call to identify the tool", "sub_task": "Identify the tool in the video"}
    {"action": "search_web", "input": "Japanese tool [tool name] Meiji era invention year", "reasoning": "Required: find historical information about the specific tool", "sub_task": "Search for tool's invention date"}
    {"action": "answer", "input": "Meiji 35", "reasoning": "All sub-tasks completed, search_web has been called, have sufficient information", "sub_task": "Final answer"}
    """

def solve_video_question(videopath,question, max_steps=20):
    """
    解决视频理解问题的主循环
    """
    print(f"\n{'='*60}")
    print(f"Starting to solve problem: {question}")
    print(f"{'='*60}\n")
    processor, model = init_vision_model()
    # Initialize conversation history
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": f"Question: {question}\n\nCRITICAL CONSTRAINTS:\n You MUST call search_web AT LEAST ONCE before providing the final answer\n\nIMPORTANT: Before calling any tool, you MUST first analyze and decompose this question into smaller, focused sub-tasks. Break down the complex question into specific sub-questions that can be answered one at a time. Then start with the first sub-task."}
    ]
    
    # 记录已观察的视频时间段
    observed_ranges = []
    step_count = 0
    video_observed = False  # 跟踪是否已调用 observe_video
    web_searched = False  # 跟踪是否已调用 search_web
    
    while step_count < max_steps:
        step_count += 1
        print(f"\n--- Step {step_count} ---")
        
        # Call model to decide next action
        response = None
        try:
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )
            
            response = completion.choices[0].message.content
            print(f"Model response: {response}")
            
            # 尝试解析JSON格式的响应
            # 首先尝试直接解析整个响应
            decision = None
            try:
                decision = json.loads(response.strip())
            except:
                # 尝试提取JSON对象
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    try:
                        decision = json.loads(json_match.group())
                    except:
                        pass
            
            # If still cannot parse, try to extract information from text
            if not decision:
                # Check if contains answer-related keywords
                if "answer" in response.lower():
                    # Try to extract answer
                    answer_match = re.search(r'(?:answer|final answer)[:：]\s*(.+)', response, re.IGNORECASE)
                    if answer_match:
                        decision = {
                            "action": "answer",
                            "input": answer_match.group(1).strip(),
                            "reasoning": "Extracted answer from response"
                        }
                    else:
                        decision = {"action": "answer", "input": response, "reasoning": "Model provided answer"}
                else:
                    # Check if tools are mentioned
                    if "observe_video" in response.lower():
                        decision = {"action": "observe_video", "input": {"question": response}, "reasoning": "Need to observe video"}
                    elif "search_web" in response.lower() or "search" in response.lower():
                        # Try to extract search query
                        query_match = re.search(r'(?:search|search_web)[:：]\s*(.+)', response, re.IGNORECASE)
                        query = query_match.group(1).strip() if query_match else response
                        decision = {"action": "search_web", "input": query, "reasoning": "Need to search for information"}
                    else:
                        # Default as reasoning
                        decision = {"action": "reason", "input": response, "reasoning": "Model returned reasoning result"}
            
            action = decision.get("action", "").lower()
            input_data = decision.get("input", "")
            reasoning = decision.get("reasoning", "")
            
            print(f"Decision: {action}")
            print(f"Reasoning: {reasoning}")
            
            # 执行相应的操作
            if action == "observe_video":
                # Check if video has already been observed
                if video_observed:
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    messages.append({
                        "role": "user",
                        "content": "ERROR: You have already called observe_video. You can ONLY call observe_video ONCE. Please use search_web to find the required information, or provide the final answer if you have sufficient information."
                    })
                    continue
                
                if isinstance(input_data, dict):
                    # time_range = input_data.get("time_range", "")
                    question = input_data.get("question", "")
                elif isinstance(input_data, str):
                    # 尝试解析字符串为JSON
                    try:
                        parsed = json.loads(input_data)
                        if isinstance(parsed, dict):
                            # time_range = parsed.get("time_range", "")
                            question = parsed.get("question", "")
                        else:
                            # time_range = ""
                            question = input_data
                    except:
                        # time_range = ""
                        question = input_data
                else:
                    # time_range = ""
                    question = str(input_data)
                
                result = observe_video(videopath,question, processor, model)
                video_observed = True  # Mark video as observed
                
                # Add to conversation history
                messages.append({
                    "role": "assistant",
                    "content": response
                })
                status_msg = f"Observation result: {result}\n\n"
                if not web_searched:
                    status_msg += "IMPORTANT: You MUST call search_web AT LEAST ONCE before providing the final answer. "
                status_msg += "Please decide the next action. You have already observed the video (can only do this once)."
                messages.append({
                    "role": "user",
                    "content": status_msg
                })
                
            elif action == "search_web":
                query = input_data if isinstance(input_data, str) else str(input_data)
                result = search_web(query)
                web_searched = True  # Mark web search as done
                
                messages.append({
                    "role": "assistant",
                    "content": response
                })
                status_msg = f"Search result: {result}\n\n"
                if not video_observed:
                    status_msg += "Note: You have not yet called observe_video. If you need to observe the video, you can call it once. "
                status_msg += "Please decide the next action. If you have all required information, you can provide the final answer."
                messages.append({
                    "role": "user",
                    "content": status_msg
                })
                
            # elif action == "read_evidence":
            #     source_id = input_data if isinstance(input_data, str) else str(input_data)
            #     result = read_evidence(source_id)
                
            #     messages.append({
            #         "role": "assistant",
            #         "content": response
            #     })
            #     messages.append({
            #         "role": "user",
            #         "content": f"Evidence content: {result}\n\nPlease decide the next action based on this result."
            #     })
                
            # elif action == "reason":
            #     input_text = input_data if isinstance(input_data, str) else str(input_data)
            #     result = reason(input_text)
                
            #     messages.append({
            #         "role": "assistant",
            #         "content": response
            #     })
            #     messages.append({
            #         "role": "user",
            #         "content": f"Reasoning result: {result}\n\nPlease decide the next action based on this result. If you already have sufficient information to answer the question, please use the answer tool."
            #     })
                
            elif action == "answer":
                # Check if search_web has been called
                if not web_searched:
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    messages.append({
                        "role": "user",
                        "content": "ERROR: You MUST call search_web AT LEAST ONCE before providing the final answer. Please use search_web to find or verify the required information first."
                    })
                    continue
                
                final_answer = input_data if isinstance(input_data, str) else str(input_data)
                print(f"\n{'='*60}")
                print(f"Final answer: {final_answer}")
                print(f"{'='*60}\n")
                return final_answer
                
            else:
                # Unknown action, treat as reasoning
                result = f"Executed action: {action}, input: {input_data}"
                messages.append({
                    "role": "assistant",
                    "content": response
                })
                messages.append({
                    "role": "user",
                    "content": f"Action result: {result}\n\nPlease continue to decide the next action."
                })
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            # If cannot parse JSON, treat response as reasoning result
            if response:
                messages.append({
                    "role": "assistant",
                    "content": response
                })
            messages.append({
                "role": "user",
                "content": "Please output your decision in JSON format: {\"action\": \"tool_name\", \"input\": \"input_parameter\", \"reasoning\": \"reason\"}"
            })
        except Exception as e:
            print(f"Execution error: {e}")
            import traceback
            traceback.print_exc()
            messages.append({
                "role": "user",
                "content": f"An error occurred: {str(e)}. Please continue."
            })
    
    print(f"\nReached maximum step limit ({max_steps}), stopping execution.")
    return None

# 主程序
if __name__ == "__main__":
    # 示例：解决单个问题
    videopath = "/mnt/sda/Datasets/VideoDR/tmp_video/2_downsampled.mp4"
    question = "The video shows the making process of a tool. According to the official description, in which year of Japan’s Meiji era was this tool invented?"
    
    answer = solve_video_question(videopath, question)