import os
import time
import numpy as np
from pdf2image import convert_from_path  # PDF转图片工具
from typing import List,Iterator # 类型注解支持
import torch
from PIL import Image
from paddleocr import PaddleOCR
import re
from langchain_core.documents import Document  # 文档对象
class ScanPDFLoader:
    """处理扫描版PDF的专用加载器，使用PaddleOCR技术提取文本"""
    
    def __init__(self):
        """
        初始化OCR引擎
        :param poppler_path: poppler工具路径(用于PDF转图片)
        """
        # 初始化PaddleOCR
        self.ocr = PaddleOCR(
            # 核心加速参数
            enable_mkldnn=True,           # 启用Intel加速库
            cpu_threads=max(8, os.cpu_count()), # 动态获取CPU核心数       
            rec_batch_num=8,              # 批量识别提高吞吐
            det_batch_num=4,              # 检测阶段批量处理
            # 精度
            use_angle_cls=True,          
            # 轻量化模型
            ocr_version='PP-OCRv4'   # 新版
        )


    def _preprocess_ocr_text(self, text: str) -> str:
        """
        单函数实现OCR文本优化：
        1. 合并错误换行（智能判断中英文语境）
        2. 修复常见OCR错误字符
        3. 规范化标点和空格
        4. 保留合理段落结构
        
        参数:
            text: OCR原始识别文本
        
        返回:
            处理后的规整文本
        """
        # 预处理：基础清洗和字符替换
        text = text.replace('\r\n', '\n').replace('\t', ' ')
        char_fixes = {'○':'。', '〈':'《', '〉':'》', '【':'[', '】':']', '…':'...', '—':'-'}
        for wrong, right in char_fixes.items():
            text = text.replace(wrong, right)
        
        # 阶段1：智能行合并
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        merged = []
        buffer = ""
        
        for line in lines:
            if not buffer:
                buffer = line
                continue
            
            # 合并规则判断
            last_char = buffer[-1]
            first_char = line[0]
            should_merge = (
                (last_char.isalpha() and first_char.isalpha()) or  # 英文单词续接
                (last_char not in {'。', '！', '？', '”', '；'} and  # 中文非句尾
                not (last_char.isupper() and first_char.isupper()))  # 非连续大写
            )
            
            if should_merge:
                connector = '' if last_char in {'“', '‘', '('} else ' '
                buffer += connector + line
            else:
                merged.append(buffer)
                buffer = line
        
        if buffer:
            merged.append(buffer)
        
        # 阶段2：空格和标点规范化
        processed_lines = []
        for line in merged:
            # 修复数字空格（1 000 → 1000）
            line = re.sub(r'(\d) (\d{3}\b)', r'\1\2', line)
            # 修复单位空格（5 kg → 5kg）
            line = re.sub(r'(\d)\s+(kg|g|ml|cm|mm|°C)\b', r'\1\2', line, flags=re.IGNORECASE)
            # 标点规范化
            line = re.sub(r'\s+([,.!?;:’”])', r'\1', line)  # 去除左空格
            line = re.sub(r'([‘“])\s+', r'\1', line)        # 去除右空格
            processed_lines.append(line)
        
        # 阶段3：生成最终文本（保留合理换行）
        result = '\n'.join(processed_lines)
        # 压缩多余空行（最多保留2个）
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result
    
    def _pdf_to_images(self, pdf_path: str) -> Iterator[Image.Image]:
        """
        流式生成PDF页面图像（带监控版）
        返回: 生成器产生PIL.Image对象，避免一次性加载所有页面
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("请先安装依赖库：pip install pymupdf pillow")

        print(f"\n[PDF解析] 开始处理文件: {os.path.basename(pdf_path)}")
        matrix = fitz.Matrix(150/72, 150/72)
        colorspace = fitz.csGRAY
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"[PDF解析] 总页数: {total_pages} | DPI: 200 | 色彩模式: 灰度")

        for page_num, page in enumerate(doc, 1):
            try:
                start_time = time.time()
                pix = page.get_pixmap(
                    matrix=matrix,
                    colorspace=colorspace,
                    alpha=False,
                    annots=False,
                    dpi=150
                )
                img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
                del pix
                
                # 打印页面处理信息
                proc_time = (time.time() - start_time) * 1000
                print(f"[PDF解析] 页面 {page_num}/{total_pages} | "
                    f"尺寸: {img.width}x{img.height} | "
                    f"耗时: {proc_time:.1f}ms", end='\r')
                
                yield img
            except Exception as e:
                print(f"\n[PDF解析] 页面 {page_num} 渲染失败: {str(e)}")
                continue
            finally:
                if 'pix' in locals():
                    del pix
        
        doc.close()
        print(f"\n[PDF解析] 所有页面渲染完成")

    def load(self, pdf_path: str) -> List[Document]:
        """
        带详细日志的PDF加载流程
        """
        import psutil
        import time

        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024 / 1024
        start_time = time.time()

        print(f"\n[OCR任务] 初始化处理 | 开始内存: {start_mem:.1f}MB")
        print("=" * 60)

        
        debug_dir = os.path.join("ocr_debug")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"存储路径: {debug_dir}")

        page_files = []
        processed_count = 0
        error_pages = []

        for i, image in enumerate(self._pdf_to_images(pdf_path), 1):
            page_file = os.path.join(debug_dir, f"page_{i}.txt")
            page_files.append(page_file)
            page_start = time.time()

            try:
                # 图像处理
                with image:
                    mem_before = process.memory_info().rss / 1024 / 1024
                    image_np = np.array(image.convert('RGB'))
                    mem_after = process.memory_info().rss / 1024 / 1024
                    print(f"\n[页面 {i}] 图像加载 | "
                        f"内存增量: {mem_after - mem_before:.1f}MB")

                # OCR处理
                ocr_start = time.time()
                result = self.ocr.ocr(image_np, cls=True)
                del image_np
                ocr_time = (time.time() - ocr_start) * 1000
                
                if result and result[0]:
                    # 文本处理
                    raw_content = "\n".join([line[1][0] for line in result[0]])
                    processed = self._preprocess_ocr_text(raw_content)
                    
                    # 写入结果
                    with open(page_file, 'w', encoding='utf-8') as f:
                        f.write(f"=== Page {i} ===\n{processed}")
                    processed_count += 1

                    # 打印统计信息
                    current_mem = process.memory_info().rss / 1024 / 1024
                    print(f"[页面 {i}] OCR完成 | "
                        f"字符数: {len(processed)} | "
                        f"OCR耗时: {ocr_time:.1f}ms | "
                        f"当前内存: {current_mem:.1f}MB")
                
            except Exception as e:
                error_pages.append(i)
                with open(page_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== Page {i} [ERROR] ===\n{str(e)}")
                print(f"\n[页面 {i}] 处理失败: {str(e)}")
                continue

        # 结果汇总
        print("\n" + "=" * 60)
        total_time = time.time() - start_time
        final_mem = process.memory_info().rss / 1024 / 1024
        mem_diff = final_mem - start_mem

        print(f"[OCR任务] 处理完成 | "
            f"总耗时: {total_time:.1f}s | "
            f"内存峰值: {final_mem:.1f}MB (+{mem_diff:.1f}MB)")
        print(f"[页面统计] 成功: {processed_count} | "
            f"失败: {len(error_pages)} | "
            f"失败页码: {error_pages if error_pages else '无'}")

        # 组装最终文本
        final_text = []
        for i, pf in enumerate(page_files, 1):
            with open(pf, 'r', encoding='utf-8') as f:
                final_text.append(f.read())

        final_output_path = os.path.join(debug_dir, "final_output.txt")
        with open(final_output_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(final_text))
      
        return [Document(
            page_content="\n\n".join(final_text),
            metadata={
                "source": pdf_path,
                "processed_pages": processed_count,
                "pages": processed_count
            }
        )]