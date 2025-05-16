import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def initialize_model():
    """모델과 토크나이저 초기화"""
    print("모델 로딩 중... 잠시만 기다려주세요.")
    model_name = "kakaocorp/kanana-nano-2.1b-instruct"
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 모델 로드 - CPU 사용 (GPU를 사용하려면 'cpu' 대신 'cuda' 지정)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cpu")
    
    model.eval()  # 평가 모드로 설정
    
    return model, tokenizer

def generate_response(model, tokenizer, conversation_history, max_tokens=256):
    """대화 기록을 바탕으로 응답 생성"""
    # 채팅 템플릿 적용
    input_ids = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 응답 생성
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # 전체 출력에서 새로 생성된 응답만 추출
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 응답 부분만 추출 (모델마다 다를 수 있음)
    # KaaNa 모델은 마지막 Assistant 응답만 가져오면 됨
    response_parts = full_response.split("Assistant: ")
    if len(response_parts) > 1:
        return response_parts[-1].strip()
    else:
        return full_response

def main():
    """대화형 채팅봇 메인 함수"""
    # 모델 초기화
    model, tokenizer = initialize_model()
    print("모델 로딩 완료! 대화를 시작합니다.")
    print("대화를 종료하려면 'exit' 또는 'quit'을 입력하세요.")
    print("-" * 50)
    
    # 대화 기록 초기화
    conversation_history = [
        {"role": "system", "content": "당신은 카카오에서 개발한 유용한 AI 어시스턴트입니다. 사용자의 질문에 친절하고 정확하게 답변해 주세요."}
    ]
    
    # 대화 루프
    while True:
        # 사용자 입력 받기
        user_input = input("\n사용자: ").strip()
        
        # 종료 조건 확인
        if user_input.lower() in ['exit', 'quit']:
            print("\n대화를 종료합니다. 감사합니다!")
            break
        
        # 사용자 메시지 추가
        conversation_history.append({"role": "user", "content": user_input})
        
        # 응답 생성 시작 표시
        print("\n어시스턴트: ", end="", flush=True)
        
        # 응답 생성
        start_time = time.time()
        response = generate_response(model, tokenizer, conversation_history)
        end_time = time.time()
        
        # 응답 출력
        print(response)
        
        # 응답 시간 정보 (선택적)
        print(f"\n(응답 생성 시간: {end_time - start_time:.2f}초)")
        
        # 어시스턴트 메시지 추가
        conversation_history.append({"role": "assistant", "content": response})
        
        # 메모리 관리를 위해 대화 기록 제한 (선택적)
        if len(conversation_history) > 11:  # 시스템 메시지 + 최대 5턴 대화 유지
            # 시스템 메시지는 유지하고 가장 오래된 사용자-어시스턴트 대화 쌍 제거
            conversation_history = [conversation_history[0]] + conversation_history[3:]

if __name__ == "__main__":
    main()