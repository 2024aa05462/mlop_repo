#!/bin/bash

# Generate PDF from Markdown report

REPORT_MD="docs/report/FINAL_REPORT.md"
REPORT_PDF="docs/report/FINAL_REPORT.pdf"

echo "📄 Generating PDF report..."

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "Pandoc could not be found. Please install it to generate PDF."
    echo "Alternative: Export to PDF using VS Code Markdown extensions."
    exit 1
fi

pandoc $REPORT_MD \
  -o $REPORT_PDF \
  --toc \
  --number-sections \
  --highlight-style=tango \
  -V geometry:margin=1in \
  -V fontsize=11pt 

if [ -f "$REPORT_PDF" ]; then
    echo "✅ PDF report generated: $REPORT_PDF"
else
    echo "❌ PDF generation failed"
    exit 1
fi
