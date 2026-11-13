"""Hierarchical Node Tree representation for documents with canonical path tags."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Node:
    """Base Node class in the document hierarchical tree."""
    id: str
    node_type: str
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[Node] = None
    children: List[Node] = field(default_factory=list)

    def add_child(self, child: Node) -> None:
        child.parent = self
        self.children.append(child)

    @property
    def canonical_path(self) -> str:
        """Returns the canonical path tag string, e.g. [sec-1/p-2/s-5]."""
        parts = []
        curr: Optional[Node] = self
        while curr is not None and curr.id != "root":
            parts.append(curr.id)
            curr = curr.parent
        parts.reverse()
        return "[" + "/".join(parts) + "]" if parts else "[root]"

    def to_raw_text(self) -> str:
        """Returns text without path annotations."""
        if not self.children:
            return self.text
        return "\n".join(c.to_raw_text() for c in self.children if c.to_raw_text().strip())

    def to_annotated_text(self) -> str:
        """Returns markdown/text annotated with bracketed canonical path tags."""
        raise NotImplementedError

    def build_index(self, index: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Builds lookup index mapping canonical paths to node text."""
        if index is None:
            index = {}
        path_tag = self.canonical_path
        # Index with and without brackets
        clean_tag = path_tag.strip("[]")
        node_text = self.text if self.text else self.to_raw_text()
        index[path_tag] = node_text
        index[clean_tag] = node_text

        for child in self.children:
            child.build_index(index)
        return index

    def find_node_by_path(self, path: str) -> Optional[Node]:
        """Find a node in this subtree matching the canonical path tag.
        
        Supports paths with or without brackets, e.g. '[sec-1/p-2/s-5]' or 'sec-1/p-2/s-5'.
        """
        clean = path.strip("[]").strip()
        if not clean:
            return None
        if clean == "root" or clean == self.id:
            return self

        parts = [p for p in clean.split("/") if p]
        if parts and parts[0] == "root" and self.id == "root":
            parts = parts[1:]
        if not parts:
            return self

        # Direct descent via child IDs
        curr = self
        for part in parts:
            matched = None
            for child in curr.children:
                if child.id == part:
                    matched = child
                    break
            if matched is None:
                # Fallback to search subtree by canonical_path
                return self._find_by_canonical_path(clean)
            curr = matched
        return curr

    def _find_by_canonical_path(self, clean_path: str) -> Optional[Node]:
        """Recursive helper matching canonical path without brackets."""
        if self.canonical_path.strip("[]") == clean_path:
            return self
        for child in self.children:
            found = child._find_by_canonical_path(clean_path)
            if found is not None:
                return found
        return None


class SectionNode(Node):
    """Section / Article / Heading level node."""
    def __init__(self, id: str, title: str = "", text: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(id=id, node_type="section", text=text, metadata=metadata or {})
        self.title = title

    def to_annotated_text(self) -> str:
        lines = []
        path_tag = self.canonical_path
        if self.title:
            lines.append(f"{path_tag} ### {self.title}")
        elif self.text:
            lines.append(f"{path_tag} {self.text}")

        for child in self.children:
            lines.append(child.to_annotated_text())
        return "\n\n".join(l for l in lines if l.strip())


class ParagraphNode(Node):
    """Paragraph level node."""
    def __init__(self, id: str, text: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(id=id, node_type="paragraph", text=text, metadata=metadata or {})

    def to_annotated_text(self) -> str:
        if not self.children:
            return f"{self.canonical_path} {self.text}"
        child_texts = [c.to_annotated_text() for c in self.children]
        return " ".join(child_texts)


class SentenceNode(Node):
    """Sentence or single statement level node."""
    def __init__(self, id: str, text: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(id=id, node_type="sentence", text=text, metadata=metadata or {})

    def to_annotated_text(self) -> str:
        return f"{self.canonical_path} {self.text}"


class TableNode(Node):
    """Table level node."""
    def __init__(self, id: str, text: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(id=id, node_type="table", text=text, metadata=metadata or {})

    def to_annotated_text(self) -> str:
        lines = [f"{self.canonical_path} [Table]"]
        for child in self.children:
            lines.append(child.to_annotated_text())
        return "\n".join(lines)


class TableRowNode(Node):
    """Table row level node."""
    def __init__(self, id: str, text: str = "", cells: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(id=id, node_type="table_row", text=text, metadata=metadata or {})
        self.cells = cells or []

    def to_annotated_text(self) -> str:
        return f"{self.canonical_path} {self.text}"


class ImageNode(Node):
    """Image or figure level node."""
    def __init__(self, id: str, caption: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(id=id, node_type="image", text=caption, metadata=metadata or {})

    def to_annotated_text(self) -> str:
        return f"{self.canonical_path} [Image: {self.text}]"


def split_into_sentences(text: str) -> List[str]:
    """Splits a paragraph into sentences or distinct lines."""
    text = text.strip()
    if not text:
        return []
    # If text has multiple bullet points or lines, split by lines
    if "\n" in text:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) > 1 and any(l.startswith(("-", "*", "•", "[", "|")) for l in lines):
            return lines

    # Sentence boundary regex
    pattern = r"(?<=[.?!;])\s+(?=[A-Z0-9\"'\(])"
    parts = re.split(pattern, text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [text]


def build_tree_from_markdown(text: str) -> SectionNode:
    """Builds a hierarchical Node Tree from markdown or raw text.
    
    Structure:
    Root Section
      -> SectionNode (sec-1, sec-2, ...)
          -> ParagraphNode (p-1, p-2, ...)
              -> SentenceNode (s-1, s-2, ...)
          -> TableNode (table-0, table-1, ...)
              -> TableRowNode (row-0, row-1, ...)
    """
    root = SectionNode(id="root", title="Document Root")

    # Check for markdown tables or sections
    raw_blocks = re.split(r"\n\s*\n", text.strip())
    sec_idx = 0
    current_sec: Optional[SectionNode] = None

    def get_or_create_section(title: str = "") -> SectionNode:
        nonlocal sec_idx, current_sec
        sec_idx += 1
        sec = SectionNode(id=f"sec-{sec_idx}", title=title)
        root.add_child(sec)
        current_sec = sec
        return sec

    table_idx = 0

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # Check if block is a heading
        heading_match = re.match(r"^(#{1,6}|ARTICLE\s+[IVXLCDM0-9]+|SECTION\s+[0-9]+|[0-9]+\.\s+[A-Z\s]{3,})\s*(.*)", block, re.IGNORECASE)
        if heading_match and len(block) < 200:
            title = block.lstrip("#").strip()
            current_sec = get_or_create_section(title)
            continue

        if current_sec is None:
            current_sec = get_or_create_section()

        # Check if block is a markdown table
        lines = block.splitlines()
        is_table = len(lines) >= 2 and all("|" in l for l in lines[:2])

        if is_table:
            t_node = TableNode(id=f"table-{table_idx}")
            table_idx += 1
            for r_idx, line in enumerate(lines):
                line_str = line.strip()
                if line_str.replace("|", "").replace("-", "").replace(":", "").strip() == "":
                    continue  # Table header separator
                row_node = TableRowNode(id=f"row-{r_idx}", text=line_str)
                t_node.add_child(row_node)
            current_sec.add_child(t_node)
            continue

        # Normal text paragraph
        p_idx = len([c for c in current_sec.children if isinstance(c, ParagraphNode)]) + 1
        p_node = ParagraphNode(id=f"p-{p_idx}", text=block)
        sentences = split_into_sentences(block)

        for s_idx, sent in enumerate(sentences, start=1):
            s_node = SentenceNode(id=f"s-{s_idx}", text=sent)
            p_node.add_child(s_node)

        current_sec.add_child(p_node)

    if not root.children:
        # Fallback for empty/single-sentence text
        sec = SectionNode(id="sec-1")
        p = ParagraphNode(id="p-1", text=text)
        p.add_child(SentenceNode(id="s-1", text=text))
        sec.add_child(p)
        root.add_child(sec)

    return root

