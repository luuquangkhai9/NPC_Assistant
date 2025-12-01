# NPC Tumor Report Generation System

Hệ thống phân tích và tạo báo cáo khối u vòm họng (Nasopharyngeal Carcinoma) sử dụng AI.

## 🌟 Tính năng

- **Phân đoạn khối u**: Sử dụng U-Net để phân đoạn GTV (Gross Tumor Volume)
- **Phân tích tự động**: Trích xuất đặc điểm khối u (thể tích, kích thước, hình thái)
- **Trực quan hóa**: Tạo hình ảnh multi-slice, 3-plane view
- **Báo cáo AI**: Sử dụng Gemini 2.0 Flash để tạo báo cáo y khoa
- **Chat tương tác**: Hỏi đáp về kết quả phân tích
- **API Backend**: FastAPI với RESTful endpoints
- **Web UI**: Gradio interface dễ sử dụng

## 📁 Cấu trúc project

```
npc_system/
├── __init__.py          # Package initialization
├── config.py            # Configuration management
├── models.py            # U-Net model & tumor analysis
├── gemini_service.py    # Gemini API integration
├── visualization.py     # Visualization generation
├── pipeline.py          # Main processing pipeline
├── api.py               # FastAPI backend
├── gradio_ui.py         # Gradio web interface
├── run.py               # Main entry point
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
└── README.md            # Documentation
```

## 🚀 Cài đặt

### 1. Clone và cài đặt dependencies

```bash
cd npc_system
pip install -r requirements.txt
```

### 2. Cấu hình

Copy file `.env.example` thành `.env` và điền API key:

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Chạy hệ thống

#### Chạy Gradio Web UI:
```bash
python run.py gradio
```

#### Chạy FastAPI Backend:
```bash
python run.py api
```

#### Chạy cả hai:
```bash
python run.py both
```

## 🐳 Chạy với Docker

```bash
# Build và chạy
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng
docker-compose down
```

## 📡 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| GET | `/cases` | Danh sách cases |
| POST | `/process` | Xử lý case |
| POST | `/process/stream` | Xử lý với streaming |
| POST | `/upload` | Upload và xử lý file |
| POST | `/chat` | Chat với AI |
| GET | `/chat/history` | Lịch sử chat |
| POST | `/chat/reset` | Reset chat session |
| GET | `/reports` | Danh sách báo cáo |
| GET | `/reports/{id}` | Chi tiết báo cáo |
| GET | `/reports/{id}/image/{type}` | Hình ảnh báo cáo |

## 🖥️ Gradio Interface

Truy cập: `http://localhost:7860`

### Tabs:
1. **📁 Xử lý Case**: Chọn hoặc upload file HDF5
2. **💬 Hỏi đáp AI**: Chat về kết quả phân tích
3. **⚙️ Cài đặt**: Cấu hình hệ thống

## 📊 API Usage Examples

### Python
```python
import requests

# Process a case
response = requests.post(
    "http://localhost:8000/process",
    json={
        "filename": "OA_CenterA_ano_set_A_005.h5",
        "dataset": "test",
        "generate_report": True
    }
)
result = response.json()
print(result['report'])

# Chat about results
response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Giải thích sphericity"}
)
print(response.json()['response'])
```

### cURL
```bash
# Process case
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"filename": "OA_CenterA_ano_set_A_005.h5", "dataset": "test"}'

# Chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Khối u có nghiêm trọng không?"}'
```

## 🔧 Configuration

Các cấu hình chính trong `config.py`:

| Config | Mô tả | Default |
|--------|-------|---------|
| `model.model_path` | Path đến U-Net model | `outputs/.../unet_best_model.pth` |
| `model.device` | Device (cuda/cpu) | `cuda` |
| `gemini.model_name` | Gemini model | `gemini-2.0-flash` |
| `gemini.temperature` | Temperature | `0.3` |
| `server.api_port` | API port | `8000` |
| `server.gradio_port` | Gradio port | `7860` |

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
