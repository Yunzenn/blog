#!/usr/bin/env python3
"""
Convert Mermaid diagrams to use Mermaid.ink URLs for cross-platform compatibility.
This works on Zhihu, Juejin, and other platforms that support external images.
"""

import re
import base64
import urllib.parse
from pathlib import Path

def mermaid_to_url(mermaid_code):
    """Convert Mermaid code to mermaid.ink URL."""
    # Encode the Mermaid code
    encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('ascii')
    # Create the URL
    return f"https://mermaid.ink/img/{encoded}"

def process_markdown_file(md_file):
    """Process a markdown file to replace Mermaid blocks with image URLs."""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)```'
    matches = list(re.finditer(pattern, content, re.DOTALL))

    if not matches:
        return False

    # Replace from end to start to maintain correct positions
    replacements = []
    for idx, match in enumerate(matches, 1):
        mermaid_code = match.group(1).strip()
        img_url = mermaid_to_url(mermaid_code)

        # Create replacement with both image and collapsible Mermaid code
        replacement = f"""![Diagram {idx}]({img_url})

<details>
<summary>查看Mermaid源码</summary>

```mermaid
{mermaid_code}
```
</details>"""

        replacements.append((match.start(), match.end(), replacement))

    # Apply replacements from end to start
    for start, end, replacement in reversed(replacements):
        content = content[:start] + replacement + content[end:]

    # Write back
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(content)

    return len(replacements)

def main():
    transformer_dir = Path('transformer')
    md_files = list(transformer_dir.glob('*.md'))

    total_converted = 0
    for md_file in md_files:
        count = process_markdown_file(md_file)
        if count:
            print(f"✓ {md_file.name}: Converted {count} diagram(s)")
            total_converted += count

    print(f"\n总计: 转换了 {total_converted} 个图表")
    print("\n说明:")
    print("- 图表现在使用 mermaid.ink 服务渲染，支持知乎、掘金等平台")
    print("- 点击'查看Mermaid源码'可以看到原始代码")
    print("- CSDN 仍然可以正常渲染折叠的 Mermaid 代码")

if __name__ == '__main__':
    main()
