"""
NPC Report Generation System - Gemini Report Generator
=======================================================
Handles AI report generation using Google Gemini API.
"""

import json
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # "user" or "model"
    content: str


class GeminiReportGenerator:
    """Generates medical reports using Gemini API with streaming support"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-3-pro-preview"):
        import google.generativeai as genai
        
        self.api_key = api_key
        self.model_name = model_name
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        self.chat_session = None
        self.chat_history: List[ChatMessage] = []
        self.current_case_context: Optional[Dict] = None
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for medical report generation"""
        return """Bạn là một chuyên gia y khoa về ung thư vòm họng (Nasopharyngeal Carcinoma - NPC).
        
Nhiệm vụ của bạn:
1. Phân tích kết quả phân đoạn khối u GTV (Gross Tumor Volume)
2. Tạo báo cáo y khoa chi tiết bằng tiếng Việt
3. Trả lời câu hỏi của bác sĩ về kết quả

Hướng dẫn:
- Sử dụng thuật ngữ y khoa chính xác
- Cung cấp phân tích khách quan dựa trên số liệu
- Đề xuất các bước tiếp theo nếu phù hợp
- Luôn nhắc nhở rằng kết quả cần được bác sĩ chuyên khoa xác nhận

Định dạng báo cáo:
1. Thông tin bệnh nhân
2. Kết quả phân đoạn
3. Phân tích đặc điểm khối u
4. Đánh giá và nhận xét
5. Khuyến nghị"""
    
    def generate_report(self, tumor_features: Dict, patient_id: str = "Unknown",
                       additional_info: str = "") -> str:
        """
        Generate a medical report for the tumor analysis.
        
        Args:
            tumor_features: Dictionary of tumor features
            patient_id: Patient identifier
            additional_info: Any additional clinical information
            
        Returns:
            Generated report text
        """
        # Store context for follow-up questions
        self.current_case_context = {
            'patient_id': patient_id,
            'tumor_features': tumor_features,
            'additional_info': additional_info
        }
        
        prompt = self._build_report_prompt(tumor_features, patient_id, additional_info)
        
        try:
            response = self.model.generate_content(prompt)
            report = response.text
            
            # Store in history
            self.chat_history.append(ChatMessage(role="user", content=f"[Yêu cầu tạo báo cáo cho {patient_id}]"))
            self.chat_history.append(ChatMessage(role="model", content=report))
            
            return report
        except Exception as e:
            return f"Lỗi khi tạo báo cáo: {str(e)}"
    
    def generate_report_stream(self, tumor_features: Dict, patient_id: str = "Unknown",
                               additional_info: str = "") -> Generator[str, None, None]:
        """
        Generate a medical report with streaming response.
        
        Yields:
            Chunks of the generated report
        """
        # Store context
        self.current_case_context = {
            'patient_id': patient_id,
            'tumor_features': tumor_features,
            'additional_info': additional_info
        }
        
        prompt = self._build_report_prompt(tumor_features, patient_id, additional_info)
        
        try:
            response = self.model.generate_content(prompt, stream=True)
            full_response = ""
            
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
            
            # Store in history after completion
            self.chat_history.append(ChatMessage(role="user", content=f"[Yêu cầu tạo báo cáo cho {patient_id}]"))
            self.chat_history.append(ChatMessage(role="model", content=full_response))
            
        except Exception as e:
            yield f"Lỗi khi tạo báo cáo: {str(e)}"
    
    def _build_report_prompt(self, tumor_features: Dict, patient_id: str, 
                            additional_info: str) -> str:
        """Build the prompt for report generation"""
        system_prompt = self._create_system_prompt()
        
        features_text = json.dumps(tumor_features, indent=2, ensure_ascii=False)
        
        prompt = f"""{system_prompt}

=== THÔNG TIN CA BỆNH ===
Mã bệnh nhân: {patient_id}
Thông tin bổ sung: {additional_info if additional_info else "Không có"}

=== KẾT QUẢ PHÂN ĐOẠN KHỐI U ===
{features_text}

=== YÊU CẦU ===
Hãy tạo báo cáo y khoa chi tiết cho ca bệnh này. Báo cáo cần bao gồm:
1. Tóm tắt kết quả phân đoạn
2. Phân tích kích thước và hình thái khối u
3. Đánh giá mức độ nghiêm trọng
4. Khuyến nghị theo dõi và điều trị
5. Lưu ý quan trọng cho bác sĩ"""
        
        return prompt
    
    def start_chat_session(self) -> None:
        """Start a new chat session for follow-up questions with images"""
        import google.generativeai as genai
        import base64
        
        # Build initial context
        initial_context = self._create_system_prompt()
        
        # Prepare message parts (text + images)
        message_parts = []
        
        if self.current_case_context:
            features = self.current_case_context.get('tumor_features', {})
            
            text_context = f"""
=== CONTEXT CA BỆNH HIỆN TẠI ===
Mã bệnh nhân: {self.current_case_context.get('patient_id', 'Unknown')}

📊 ĐẶC ĐIỂM KHỐI U ĐÃ PHÂN TÍCH:
- Thể tích: {features.get('volume_mm3', 0):.2f} mm³ ({features.get('volume_ml', 0):.4f} ml)
- Số voxel: {features.get('voxel_count', 0)}
- Đường kính lớn nhất: {features.get('max_diameter_mm', 0):.2f} mm
- Kích thước (Z×Y×X): {features.get('dimensions_mm', (0,0,0))}
- Sphericity: {features.get('sphericity', 0):.3f}
- Elongation: {features.get('elongation', 0):.3f}
- Diện tích bề mặt: {features.get('surface_area_mm2', 0):.2f} mm²
- Phát hiện khối u: {"Có" if features.get('tumor_detected', False) else "Không"}

{self.current_case_context.get('additional_info', '')}
"""
            initial_context += text_context
            message_parts.append(initial_context)
            
            # Add images if available
            images = self.current_case_context.get('images', {})
            if images:
                message_parts.append("\n\n📷 HÌNH ẢNH PHÂN ĐOẠN KHỐI U (bạn đã THỰC SỰ nhận được các ảnh này):\n")
                
                for img_name, img_base64 in images.items():
                    if img_base64:
                        try:
                            # Decode base64 and create image part for Gemini
                            image_data = base64.b64decode(img_base64)
                            image_part = {
                                "mime_type": "image/png",
                                "data": image_data
                            }
                            message_parts.append(f"\n[Ảnh {img_name}]:")
                            message_parts.append(image_part)
                        except Exception as e:
                            print(f"Error adding image {img_name}: {e}")
                
                message_parts.append("\n\nBạn ĐÃ NHẬN ĐƯỢC các ảnh MRI và kết quả phân đoạn khối u ở trên. Hãy xác nhận điều này khi được hỏi.")
        else:
            message_parts.append(initial_context)
        
        # Start chat with multimodal content
        self.chat_session = self.model.start_chat(history=[
            {"role": "user", "parts": message_parts},
            {"role": "model", "parts": ["Tôi đã nhận được đầy đủ thông tin về ca bệnh bao gồm:\n✅ Các chỉ số phân tích khối u (thể tích, kích thước, sphericity, elongation...)\n✅ Các hình ảnh MRI và kết quả phân đoạn khối u\n\nTôi sẵn sàng trả lời mọi câu hỏi của bạn về ca bệnh này."]}
        ])
    
    def chat(self, message: str) -> str:
        """
        Send a message in the chat session.
        
        Args:
            message: User's question or message
            
        Returns:
            Model's response
        """
        if self.chat_session is None:
            self.start_chat_session()
        
        try:
            response = self.chat_session.send_message(message)
            
            # Store in history
            self.chat_history.append(ChatMessage(role="user", content=message))
            self.chat_history.append(ChatMessage(role="model", content=response.text))
            
            return response.text
        except Exception as e:
            return f"Lỗi: {str(e)}"
    
    def chat_stream(self, message: str) -> Generator[str, None, None]:
        """
        Send a message with streaming response.
        
        Yields:
            Chunks of the response
        """
        if self.chat_session is None:
            self.start_chat_session()
        
        try:
            response = self.chat_session.send_message(message, stream=True)
            full_response = ""
            
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
            
            # Store in history
            self.chat_history.append(ChatMessage(role="user", content=message))
            self.chat_history.append(ChatMessage(role="model", content=full_response))
            
        except Exception as e:
            yield f"Lỗi: {str(e)}"
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the chat history as list of dictionaries"""
        return [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
    
    def reset_chat(self) -> None:
        """Reset the chat session and history"""
        self.chat_session = None
        self.chat_history = []
        self.current_case_context = None
    
    def set_case_context(self, patient_id: str, tumor_features: Dict, 
                        additional_info: str = "", 
                        images: Optional[Dict[str, str]] = None) -> None:
        """
        Set context for a new case including images.
        
        Args:
            patient_id: Patient identifier
            tumor_features: Dictionary of tumor features
            additional_info: Additional text info
            images: Dictionary of base64 encoded images {'summary': '...', 'multi_slice': '...'}
        """
        self.current_case_context = {
            'patient_id': patient_id,
            'tumor_features': tumor_features,
            'additional_info': additional_info,
            'images': images or {}
        }
        # Reset chat session to incorporate new context
        self.chat_session = None
