"""
Export Manager - Excel/PDF Export
Tuần 9: Local Database & Reporting
Export báo cáo ra Excel và PDF
"""

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import json


class ExportManager:
    """
    Export Manager for Excel and PDF reports
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize Export Manager
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_excel(
        self,
        data: List[Dict],
        filename: Optional[str] = None,
        sheet_name: str = "Customer Data"
    ) -> str:
        """
        Export data to Excel file
        
        Args:
            data: List of dictionaries with customer data
            filename: Output filename (auto-generated if None)
            sheet_name: Sheet name
            
        Returns:
            Path to exported file
        """
        if not data:
            raise ValueError("No data to export")
        
        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"customer_report_{timestamp}.xlsx"
        
        filepath = self.output_dir / filename
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Write to Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                )
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
        
        return str(filepath)
    
    def export_to_pdf(
        self,
        data: List[Dict],
        title: str = "Customer Analytics Report",
        filename: Optional[str] = None
    ) -> str:
        """
        Export data to PDF file
        
        Args:
            data: List of dictionaries with customer data
            title: Report title
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if not data:
            raise ValueError("No data to export")
        
        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"customer_report_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        
        # Create PDF
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#0078D7'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Summary
        summary_style = ParagraphStyle(
            'Summary',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12
        )
        
        total_customers = len(data)
        story.append(Paragraph(f"<b>Total Customers:</b> {total_customers}", summary_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Convert data to table
        if data:
            # Get headers
            headers = list(data[0].keys())
            
            # Prepare table data
            table_data = [headers]
            for row in data:
                table_data.append([str(row.get(h, '')) for h in headers])
            
            # Create table
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0078D7')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            story.append(table)
        
        # Footer
        story.append(Spacer(1, 0.2*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            footer_style
        ))
        
        # Build PDF
        doc.build(story)
        
        return str(filepath)
    
    def export_summary_report(
        self,
        summary_data: Dict,
        filename: Optional[str] = None
    ) -> str:
        """
        Export summary statistics to PDF
        
        Args:
            summary_data: Dictionary with summary statistics
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_report_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        story = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#0078D7'),
            alignment=TA_CENTER
        )
        
        story.append(Paragraph("Analytics Summary Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Summary items
        normal_style = styles['Normal']
        for key, value in summary_data.items():
            story.append(Paragraph(f"<b>{key}:</b> {value}", normal_style))
            story.append(Spacer(1, 0.1*inch))
        
        doc.build(story)
        
        return str(filepath)

