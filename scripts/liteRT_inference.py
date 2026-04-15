import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


# ✅ 1. 모델 불러오기
interpreter = Interpreter(model_path="./simple_dnn.tflite")
interpreter.allocate_tensors()

# ✅ 2. 입력/출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("입력 정보:", input_details)
print("출력 정보:", output_details)

# ✅ 3. 입력 데이터 준비 (예: 랜덤 테스트 입력)
input_shape = input_details[0]['shape']
dummy_input = np.random.rand(*input_shape).astype(np.float32)  # (1, 40, 448)

# ✅ 4. 추론 실행
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print("🔎 추론 결과 shape:", output_data.shape)
print("🔎 예시 출력 값:", output_data)
