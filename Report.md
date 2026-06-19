# Báo cáo kết quả và Phân tích hệ thống Memory (Day 17 - Live Mode với OpenRouter)

Báo cáo này được thực hiện dựa trên kết quả chạy **Live Mode** thực tế sử dụng API của **OpenRouter** và mô hình `mistralai/mistral-small-24b-instruct-2501`.

---

## 1. Kết quả Benchmark thực tế (Live Mode)

### Standard Benchmark (10 Conversations)
*Môi trường chạy: Live API OpenRouter, model `mistral-small-24b-instruct-2501`*

| Agent Name | Agent Tokens Only | Prompt Tokens Processed | Cross-Session Recall | Response Quality | Memory Growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 41,666 | 159,565 | 10.71% | 10.00% | 0 | 0 |
| **Advanced Agent** | 10,482 | 47,198 | **89.29%** | **89.29%** | 303 | 20 |

### Long-Context Stress Benchmark (1 Conversation - Long turns)
*Môi trường chạy: Live API OpenRouter, model `mistral-small-24b-instruct-2501`*

| Agent Name | Agent Tokens Only | Prompt Tokens Processed | Cross-Session Recall | Response Quality | Memory Growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 8,364 | 69,573 | 0.00% | 0.00% | 0 | 0 |
| **Advanced Agent** | 6,950 | 12,189 | **100.00%** | **100.00%** | 184 | 30 |

---

## 2. Phân tích Kỹ thuật chi tiết

### 2.1. Vì sao Advanced Agent có Recall tốt hơn hẳn Baseline Agent?
- **Baseline Agent** chỉ sử dụng bộ nhớ ngắn hạn (`within-session memory`). Khi bước vào một thread mới để hỏi các câu hỏi recall chéo phiên (`cross-session recall`), Baseline Agent bắt đầu từ một lịch sử hội thoại rỗng, do đó hầu như quên sạch các facts (tên, nơi ở, món ăn...) đã được cung cấp trước đó (Recall chỉ đạt **10.71%** ở bản Standard và **0.00%** ở bản Stress).
- **Advanced Agent** tích hợp lớp Persistent Memory lưu trữ hồ sơ bền vững trên đĩa dưới dạng tệp `User.md`. Tác nhân tự động cập nhật tệp này qua mỗi lượt chat. Khi chuyển sang một thread mới, tệp `User.md` được nạp trực tiếp vào prompt ngữ cảnh hệ thống giúp mô hình ghi nhớ chính xác thông tin người dùng ngay cả khi lịch sử chat ngắn hạn trống trơn (Recall đạt **89.29%** ở Standard và **100.00%** ở Stress).

### 2.2. Đánh giá chi phí Token giữa Advanced và Baseline
- **Tiết kiệm Agent Generation Tokens (`Agent Tokens Only`)**:
  - Ở Standard: Advanced chỉ dùng **10,482 tokens** so với **41,666 tokens** của Baseline.
  - Ở Stress: Advanced dùng **6,950 tokens** so với **8,364 tokens** của Baseline.
  - **Lý do**: Khi ngữ cảnh chat quá dài, Baseline Agent bị nhiễu thông tin dẫn đến việc sinh các câu trả lời dài dòng, lan man, lặp ý. Trong khi đó, Advanced Agent duy trì được phong cách trả lời ngắn gọn theo preference của người dùng (ví dụ: style trả lời ngắn, 3 bullet) nên tiết kiệm lượng token sinh ra đáng kể.
- **Tiết kiệm Prompt Tokens (`Prompt Tokens Processed`)**:
  - Ở Standard: Advanced chỉ xử lý **47,198 tokens** prompt so với **159,565 tokens** của Baseline (giảm 70.4%).
  - Ở Stress: Advanced chỉ xử lý **12,189 tokens** prompt so với **69,573 tokens** của Baseline (giảm 82.5%).
  - **Lý do**: Nhờ thuật toán Compact Memory quản lý lịch sử hội thoại. Khi độ dài lịch sử hội thoại vượt ngưỡng threshold, lớp Compact Memory sẽ kích hoạt nén (Compactions = 20 và 30 lần). Lớp này nén các tin nhắn cũ hơn thành dạng summary ngắn và giải phóng các tin nhắn cũ ra khỏi prompt context, tránh làm phồng prompt theo cấp số cộng O(N^2) như Baseline.

### 2.3. Sự tăng trưởng của Memory File (`User.md`) và Rủi ro đi kèm
- **Tốc độ tăng trưởng**: Tệp profile tăng trưởng rất chậm và ổn định (chỉ tăng **303 bytes** ở Standard và **184 bytes** ở Stress), bởi vì chỉ lưu các fact tĩnh, cô đọng của người dùng (tên, sở thích, nơi ở hiện tại...).
- **Rủi ro đi kèm**:
  1. *Xung đột thông tin (Conflict/Outdated facts)*: Nếu người dùng đính chính nơi ở (từ Huế sang Đà Nẵng) hoặc công việc (từ backend sang MLOps), nếu không có cơ chế `edit_text` thay thế dòng cũ mà chỉ append thì file sẽ chứa cả thông tin cũ lẫn mới, gây bối rối cho LLM. Tác nhân Advanced Agent của chúng ta đã khắc phục điều này bằng cách sử dụng `edit_text` để thay thế trực tiếp dòng thông tin cũ.
  2. *Ghi nhận thông tin nhiễu*: Khi người dùng đưa ra câu đùa ("đùa với đồng nghiệp là chuyển sang Product Manager") hoặc các thông tin tham chiếu không phải của mình ("Hà Nội chỉ là nơi mình bay ra họp"), nếu bộ trích xuất fact không lọc nhiễu tốt sẽ ghi nhận sai thông tin. Việc sử dụng các heuristic lọc câu hỏi/câu phủ định đã giúp tác nhân đạt recall 100% trong Stress test.
