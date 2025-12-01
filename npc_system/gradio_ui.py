"""
NPC Report Generation System - Gradio Web UI
=============================================
Interactive web interface for the NPC report generation system.
"""

import os
import json
import base64
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import tempfile

import gradio as gr

from .config import get_config, update_gemini_api_key
from .pipeline import NPCReportPipeline


# ============================================================
# Global State
# ============================================================

pipeline: Optional[NPCReportPipeline] = None
chat_history: List[Dict[str, str]] = []  # Gradio 6.0 uses messages format


def get_pipeline() -> NPCReportPipeline:
    """Get or create the pipeline instance"""
    global pipeline
    if pipeline is None:
        pipeline = NPCReportPipeline()
    return pipeline


def initialize_system(api_key: str) -> str:
    """Initialize the system with API key"""
    global pipeline
    
    if not api_key.strip():
        return "❌ Vui lòng nhập Gemini API Key"
    
    try:
        update_gemini_api_key(api_key.strip())
        pipeline = NPCReportPipeline()
        success = pipeline.initialize()
        
        if success:
            return "✅ Hệ thống đã khởi tạo thành công!"
        else:
            return "❌ Lỗi khởi tạo. Kiểm tra model path."
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"


def get_available_cases() -> gr.Dropdown:
    """Get list of available cases and return updated dropdown"""
    p = get_pipeline()
    if not p.is_initialized:
        return gr.Dropdown(choices=[], value=None)
    
    cases = p.list_available_cases()
    all_cases = []
    
    for name in cases.get('test', []):
        all_cases.append(f"[TEST] {name}")
    for name in cases.get('val', []):
        all_cases.append(f"[VAL] {name}")
    
    return gr.Dropdown(choices=all_cases, value=None)


def process_selected_case(case_selection: str, progress=gr.Progress()) -> Tuple:
    """Process a selected case from the dropdown"""
    global chat_history
    chat_history = []
    
    if not case_selection:
        return None, None, None, "", "❌ Chọn một case để xử lý", []
    
    p = get_pipeline()
    if not p.is_initialized:
        return None, None, None, "", "❌ Hệ thống chưa được khởi tạo", []
    
    # Parse selection
    if case_selection.startswith("[TEST]"):
        dataset = "test"
        filename = case_selection.replace("[TEST] ", "")
    else:
        dataset = "val"
        filename = case_selection.replace("[VAL] ", "")
    
    case_path = p.get_case_path(filename, dataset)
    if not case_path:
        return None, None, None, "", f"❌ Không tìm thấy file: {filename}", []
    
    # Process with progress updates
    results = {}
    features_text = ""
    report_text = ""
    patient_id = ""
    
    try:
        progress(0.1, desc="Đang tải dữ liệu...")
        
        for update in p.process_case_stream(case_path):
            step = update.get('step', '')
            msg = update.get('message', '')
            prog = update.get('progress', 0) / 100
            
            progress(prog, desc=msg)
            
            if step == 'loaded':
                patient_id = update.get('message', '').replace('Đã tải: ', '')
            
            if step == 'analyzed':
                results['features'] = update.get('features', {})
                features_text = format_features(results['features'])
            
            if step == 'visualized':
                results['visualizations'] = update.get('visualizations', {})
            
            if step == 'report_chunk':
                report_text += update.get('chunk', '')
            
            if step == 'reported':
                report_text = update.get('report', report_text)
            
            if step == 'completed':
                patient_id = update.get('patient_id', patient_id)
            
            if step == 'error':
                return None, None, None, "", f"❌ {update.get('error', 'Unknown error')}", []
        
        # Decode images
        img_multi = decode_base64_image(results.get('visualizations', {}).get('multi_slice', ''))
        img_3plane = decode_base64_image(results.get('visualizations', {}).get('three_plane', ''))
        img_summary = decode_base64_image(results.get('visualizations', {}).get('summary', ''))
        
        # === TẠO CHAT HISTORY VỚI BÁO CÁO ===
        # Khởi tạo chat với context đầy đủ về case vừa xử lý
        chat_history = [
            {
                "role": "assistant", 
                "content": f"""🏥 **ĐÃ HOÀN THÀNH PHÂN TÍCH CA BỆNH: {patient_id}**

Tôi đã nhận được đầy đủ thông tin về ca bệnh này bao gồm:
- ✅ Ảnh MRI và kết quả phân đoạn khối u
- ✅ Các chỉ số đặc điểm khối u (thể tích, kích thước, hình thái)
- ✅ Hình ảnh trực quan hóa (multi-slice, 3-plane view)

---

{report_text}

---

💬 **Bạn có thể hỏi tôi bất kỳ câu hỏi nào về:**
- Ý nghĩa các chỉ số (sphericity, elongation, thể tích...)
- Đánh giá mức độ nghiêm trọng của khối u
- Khuyến nghị theo dõi và điều trị
- So sánh với các ca tương tự
- Giải thích chi tiết về báo cáo"""
            }
        ]
        
        # Đảm bảo Gemini service có context đầy đủ VỚI ẢNH
        if p._gemini:
            # Lấy ảnh base64 để gửi vào Gemini
            images_for_gemini = {
                'summary': results.get('visualizations', {}).get('summary', ''),
                'multi_slice': results.get('visualizations', {}).get('multi_slice', ''),
                'three_plane': results.get('visualizations', {}).get('three_plane', '')
            }
            p._gemini.set_case_context(
                patient_id=patient_id,
                tumor_features=results.get('features', {}),
                additional_info=f"Báo cáo đã tạo: {report_text[:500]}...",
                images=images_for_gemini  # Gửi ảnh thực tế
            )
        
        return img_multi, img_3plane, img_summary, features_text, report_text, chat_history
        
    except Exception as e:
        return None, None, None, "", f"❌ Lỗi xử lý: {str(e)}", []


def process_uploaded_file(file, progress=gr.Progress()) -> Tuple:
    """Process an uploaded HDF5 file"""
    global chat_history
    chat_history = []
    
    if file is None:
        return None, None, None, "", "❌ Chọn file để upload", []
    
    p = get_pipeline()
    if not p.is_initialized:
        return None, None, None, "", "❌ Hệ thống chưa được khởi tạo", []
    
    try:
        # Process uploaded file
        file_path = Path(file.name)
        
        results = {}
        features_text = ""
        report_text = ""
        patient_id = ""
        
        progress(0.1, desc="Đang tải dữ liệu...")
        
        for update in p.process_case_stream(file_path):
            step = update.get('step', '')
            msg = update.get('message', '')
            prog = update.get('progress', 0) / 100
            
            progress(prog, desc=msg)
            
            if step == 'loaded':
                patient_id = update.get('message', '').replace('Đã tải: ', '')
            
            if step == 'analyzed':
                results['features'] = update.get('features', {})
                features_text = format_features(results['features'])
            
            if step == 'visualized':
                results['visualizations'] = update.get('visualizations', {})
            
            if step == 'report_chunk':
                report_text += update.get('chunk', '')
            
            if step == 'reported':
                report_text = update.get('report', report_text)
            
            if step == 'completed':
                patient_id = update.get('patient_id', patient_id)
            
            if step == 'error':
                return None, None, None, "", f"❌ {update.get('error', 'Unknown error')}", []
        
        # Decode images
        img_multi = decode_base64_image(results.get('visualizations', {}).get('multi_slice', ''))
        img_3plane = decode_base64_image(results.get('visualizations', {}).get('three_plane', ''))
        img_summary = decode_base64_image(results.get('visualizations', {}).get('summary', ''))
        
        # === TẠO CHAT HISTORY VỚI BÁO CÁO ===
        chat_history = [
            {
                "role": "assistant", 
                "content": f"""🏥 **ĐÃ HOÀN THÀNH PHÂN TÍCH CA BỆNH: {patient_id}**

Tôi đã nhận được đầy đủ thông tin về ca bệnh này bao gồm:
- ✅ Ảnh MRI và kết quả phân đoạn khối u
- ✅ Các chỉ số đặc điểm khối u (thể tích, kích thước, hình thái)
- ✅ Hình ảnh trực quan hóa (multi-slice, 3-plane view)

---

{report_text}

---

💬 **Bạn có thể hỏi tôi bất kỳ câu hỏi nào về:**
- Ý nghĩa các chỉ số (sphericity, elongation, thể tích...)
- Đánh giá mức độ nghiêm trọng của khối u
- Khuyến nghị theo dõi và điều trị
- So sánh với các ca tương tự
- Giải thích chi tiết về báo cáo"""
            }
        ]
        
        # Đảm bảo Gemini service có context đầy đủ VỚI ẢNH
        if p._gemini:
            # Lấy ảnh base64 để gửi vào Gemini
            images_for_gemini = {
                'summary': results.get('visualizations', {}).get('summary', ''),
                'multi_slice': results.get('visualizations', {}).get('multi_slice', ''),
                'three_plane': results.get('visualizations', {}).get('three_plane', '')
            }
            p._gemini.set_case_context(
                patient_id=patient_id,
                tumor_features=results.get('features', {}),
                additional_info=f"Báo cáo đã tạo: {report_text[:500]}...",
                images=images_for_gemini  # Gửi ảnh thực tế
            )
        
        return img_multi, img_3plane, img_summary, features_text, report_text, chat_history
        
    except Exception as e:
        return None, None, None, "", f"❌ Lỗi: {str(e)}", []


def chat_with_ai(message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """Handle chat interaction - Gradio 6.0 messages format"""
    global chat_history
    
    if not message.strip():
        return "", history
    
    p = get_pipeline()
    if not p.is_initialized:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ Hệ thống chưa được khởi tạo"})
        return "", history
    
    if not p.current_case:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ Chưa có case nào được xử lý. Vui lòng xử lý một case trước."})
        return "", history
    
    try:
        # Get streaming response
        response = ""
        for chunk in p.chat(message, stream=True):
            response += chunk
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        chat_history = history
        return "", history
        
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Lỗi: {str(e)}"})
        return "", history


def reset_chat_history() -> List[Dict[str, str]]:
    """Reset chat history"""
    global chat_history
    chat_history = []
    
    p = get_pipeline()
    if p and p.is_initialized:
        p.reset_chat()
    
    return []


def format_features(features: dict) -> str:
    """Format features for display"""
    if not features:
        return "Không có dữ liệu"
    
    text = """
📊 **ĐẶC ĐIỂM KHỐI U**

🔬 **Thể tích:**
- Volume: {volume_mm3:.2f} mm³ ({volume_ml:.4f} ml)
- Số voxel: {voxel_count}

📏 **Kích thước:**
- Đường kính lớn nhất: {max_diameter_mm:.2f} mm
- Kích thước (Z×Y×X): {dim_z:.1f} × {dim_y:.1f} × {dim_x:.1f} mm

🔵 **Hình thái:**
- Sphericity: {sphericity:.3f}
- Elongation: {elongation:.3f}
- Diện tích bề mặt: {surface_area_mm2:.2f} mm²

✅ **Phát hiện khối u:** {tumor_detected}
""".format(
        volume_mm3=features.get('volume_mm3', 0),
        volume_ml=features.get('volume_ml', 0),
        voxel_count=features.get('voxel_count', 0),
        max_diameter_mm=features.get('max_diameter_mm', 0),
        dim_z=features.get('dimensions_mm', (0,0,0))[0],
        dim_y=features.get('dimensions_mm', (0,0,0))[1],
        dim_x=features.get('dimensions_mm', (0,0,0))[2],
        sphericity=features.get('sphericity', 0),
        elongation=features.get('elongation', 0),
        surface_area_mm2=features.get('surface_area_mm2', 0),
        tumor_detected="Có" if features.get('tumor_detected', False) else "Không"
    )
    return text


def decode_base64_image(base64_str: str):
    """Decode base64 string to image"""
    if not base64_str:
        return None
    
    try:
        import io
        from PIL import Image
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception:
        return None


def create_gradio_app() -> gr.Blocks:
    """Create the Gradio interface"""
    
    with gr.Blocks(title="NPC Tumor Report Generation") as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🏥 NPC Tumor Report Generation System</h1>
            <p>Hệ thống phân tích và tạo báo cáo khối u vòm họng sử dụng AI</p>
        </div>
        """)
        
        # Initialization section
        with gr.Row():
            with gr.Column(scale=3):
                api_key_input = gr.Textbox(
                    label="Gemini API Key",
                    placeholder="Nhập API key của bạn...",
                    type="password"
                )
            with gr.Column(scale=1):
                init_btn = gr.Button("🚀 Khởi tạo hệ thống", variant="primary")
            with gr.Column(scale=2):
                init_status = gr.Textbox(label="Trạng thái", interactive=False)
        
        gr.Markdown("---")
        
        # Main tabs
        with gr.Tabs():
            
            # Tab 1: Process Cases
            with gr.Tab("📁 Xử lý Case"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Chọn case có sẵn")
                        case_dropdown = gr.Dropdown(
                            label="Case",
                            choices=[],
                            interactive=True,
                            allow_custom_value=True  # Fix warning
                        )
                        refresh_btn = gr.Button("🔄 Làm mới danh sách")
                        process_btn = gr.Button("▶️ Xử lý case", variant="primary")
                        
                        gr.Markdown("### Hoặc upload file")
                        file_upload = gr.File(
                            label="Upload HDF5 file",
                            file_types=[".h5"]
                        )
                        upload_btn = gr.Button("📤 Upload và xử lý")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Kết quả phân tích")
                        features_display = gr.Markdown(label="Đặc điểm khối u")
                
                gr.Markdown("### 📊 Hình ảnh trực quan")
                with gr.Row():
                    img_multi = gr.Image(label="Multi-Slice View", type="pil")
                    img_3plane = gr.Image(label="3-Plane View", type="pil")
                
                with gr.Row():
                    img_summary = gr.Image(label="Summary Figure", type="pil")
                
                gr.Markdown("### 📝 Báo cáo AI")
                report_display = gr.Markdown(label="Báo cáo")
            
            # Tab 2: Chat
            with gr.Tab("💬 Hỏi đáp AI"):
                gr.Markdown("""
                ### Chat với AI về kết quả phân tích
                
                Bạn có thể hỏi các câu hỏi như:
                - *"Giải thích ý nghĩa của sphericity"*
                - *"Khối u này có nghiêm trọng không?"*
                - *"Cần theo dõi như thế nào?"*
                """)
                
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=400
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Nhập câu hỏi",
                        placeholder="Hỏi về kết quả phân tích...",
                        scale=4 
                    )
                    chat_btn = gr.Button("Gửi", variant="primary", scale=1)
                
                clear_chat_btn = gr.Button("🗑️ Xóa lịch sử chat")
            
            # Tab 3: Settings
            with gr.Tab("⚙️ Cài đặt"):
                gr.Markdown("### Cấu hình hệ thống")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        **Thông tin hệ thống:**
                        - Model: U-Net cho phân đoạn GTV
                        - AI: Gemini 2.0 Flash
                        - Định dạng input: HDF5
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        **Hướng dẫn sử dụng:**
                        1. Nhập Gemini API Key và khởi tạo
                        2. Chọn case hoặc upload file
                        3. Xem kết quả và báo cáo
                        4. Chat để hỏi thêm về kết quả
                        """)
        
        # Event handlers
        init_btn.click(
            fn=initialize_system,
            inputs=[api_key_input],
            outputs=[init_status]
        ).then(
            fn=get_available_cases,
            outputs=[case_dropdown]
        )
        
        refresh_btn.click(
            fn=get_available_cases,
            outputs=[case_dropdown]
        )
        
        process_btn.click(
            fn=process_selected_case,
            inputs=[case_dropdown],
            outputs=[img_multi, img_3plane, img_summary, features_display, report_display, chatbot]
        )
        
        upload_btn.click(
            fn=process_uploaded_file,
            inputs=[file_upload],
            outputs=[img_multi, img_3plane, img_summary, features_display, report_display, chatbot]
        )
        
        # Chat handlers
        chat_btn.click(
            fn=chat_with_ai,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot]
        )
        
        chat_input.submit(
            fn=chat_with_ai,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot]
        )
        
        clear_chat_btn.click(
            fn=reset_chat_history,
            outputs=[chatbot]
        )
    
    return app


def launch_gradio(share: bool = False, server_port: int = 7860):
    """Launch the Gradio app"""
    app = create_gradio_app()
    
    print(f"\n{'='*60}")
    print("🏥 NPC Tumor Report Generation System")
    print(f"{'='*60}")
    print(f"🌐 Truy cập tại: http://localhost:{server_port}")
    print(f"   hoặc: http://127.0.0.1:{server_port}")
    print(f"{'='*60}\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    launch_gradio()
