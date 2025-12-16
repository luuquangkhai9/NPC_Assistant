# NPC Tumor Report Generation System

Hệ thống phân tích và tạo báo cáo khối u vòm họng (Nasopharyngeal Carcinoma) sử dụng AI.

## 🌟 Tính năng

- **Phân đoạn khối u**: Sử dụng U-Net để phân đoạn GTV (Gross Tumor Volume)
- **Phân tích tự động**: Trích xuất đặc điểm khối u (thể tích, kích thước, hình thái)
- **Trực quan hóa**: Tạo hình ảnh multi-slice, 3-plane view
- **Báo cáo AI**: Sử dụng Gemini 3 pro để tạo báo cáo y khoa
- **Chat tương tác**: Hỏi đáp về kết quả phân tích
- **API Backend**: FastAPI với RESTful endpoints
- **Web UI**: Gradio interface dễ sử dụng

## 📁 Cấu trúc project

```
.
├── convert_data.py      # Script chuyển đổi dữ liệu
├── training_pipeline.ipynb # Notebook huấn luyện U-Net
├── trainning_swinunet.ipynb # Notebook huấn luyện Swin-UNet
├── npc_system/          # Package source code
│   ├── __init__.py      # Package initialization
│   ├── config.py        # Configuration management
│   ├── models.py        # U-Net model & tumor analysis
│   ├── gemini_service.py # Gemini API integration
│   ├── visualization.py # Visualization generation
│   ├── pipeline.py      # Main processing pipeline
│   ├── api.py           # FastAPI backend
│   ├── gradio_ui.py     # Gradio web interface
│   ├── run.py           # Main entry point
│   ├── requirements.txt # Python dependencies
│   ├── Dockerfile       # Docker configuration
│   └── docker-compose.yml # Docker Compose setup
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

<!-- uvicorn api:app --host 0.0.0.0 --port 8000 --reload -->

#### Chạy cả hai:
```bash
python run.py both
```

## �️ Chuẩn bị dữ liệu & Training
### 0. Tải dữ liệu
Dữ liệu sử dụng trong dự án được cung cấp bởi bài báo [SFADA-GTV-Seg](https://www.redjournal.org/article/S0360-3016(24)03644-7/fulltext).

Bạn có thể tải dataset từ Google Drive:
- **Link tải**: [Google Drive Folder](https://drive.google.com/drive/folders/1y7GNrIqkqzebyJhaO-G1rDvs1OhCVYgh)

Sau khi tải về, hãy giải nén và sắp xếp thư mục như hướng dẫn bên dưới.
### 1. Cấu trúc dữ liệu đầu vào
Dữ liệu NIfTI cần được tổ chức theo cấu trúc sau:
```
dataset_root/
├── CenterA/
│   ├── images/
│   │   ├── case001.nii.gz
│   │   └── ...
│   └── labels/
│       ├── case001.nii.gz
│       └── ...
├── CenterB/
│   └── ...
```

### 2. Chuyển đổi dữ liệu
Sử dụng script `convert_data.py` để chuyển đổi dữ liệu NIfTI sang định dạng `.npz` (cho training) và `.h5` (cho validation/test).

```bash
# Cài đặt thư viện cần thiết (nếu chưa có)
pip install h5py SimpleITK scipy tqdm scikit-image

# Chạy chuyển đổi
python convert_data.py --dataset-root /path/to/your/dataset --output-root ./outputs/swin_dataset_npz
```

Các tham số tùy chọn:
- `--img-size`: Kích thước ảnh (mặc định 224)
- `--force`: Bắt buộc chuyển đổi lại từ đầu

### 3. Huấn luyện mô hình
Sử dụng các notebook trong thư mục gốc để huấn luyện:
- `trainning_swinunet.ipynb`: Huấn luyện Swin-UNet (khuyên dùng)
- `training_pipeline.ipynb`: Huấn luyện U-Net cơ bản

## �🐳 Chạy với Docker

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
| Path đến Swin U-Net model | `outputs/.../swin_best_model.pth` |
| `model.device` | Device (cuda/cpu) | `cuda` |
| `gemini.model_name` | Gemini model | `gemini-3-pro-preview` |
| `gemini.temperature` | Temperature | `0.3` |
| `server.api_port` | API port | `8080` |
| `server.gradio_port` | Gradio port | `7860` |

