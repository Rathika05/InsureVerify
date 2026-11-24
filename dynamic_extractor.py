"""Discover printed labels and adjacent values without a fixed field list."""
import re
from spatial_extractor import build_rows, words_box

def norm(s):
    return re.sub(r'[^a-z0-9]+',' ',str(s).lower()).strip()

def label_ok(s):
    return bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9 /&().,%_-]{1,85}',s)) and 1 <= len(s.split()) <= 10 and not re.search(r'\b(?:please|shall|hereby|signature|seal|http|www)\b',s,re.I)

def value_ok(s):
    return bool(s and len(s)<=350 and norm(s) not in {'','na','n a','nil','not applicable'} and re.search(r'\w',s))

def discover(payload, schema):
    known={norm(a) for f, aliases in schema.items() for a in [f,*aliases]}
    seen=set(); out=[]
    def emit(page,label,value,box,evidence):
        label=label.strip(' :'); value=value.strip(' :|')
        key=(norm(label),norm(value))
        if not label_ok(label) or not value_ok(value) or norm(label) in known or norm(value) in known or key in seen or not box: return
        if norm(label) in {'name','address','phone','email','contact number','gstin','pan','date','total','amount'}: return
        if len(box)!=4 or not (0<=box[0]<box[2]<=1000 and 0<=box[1]<box[3]<=1000): return
        seen.add(key)
        out.append(dict(field=label,value=value,page=page.get('page'),bbox=list(box),status='Needs review',evidence='Dynamic label; verify meaning | '+evidence[:350]))
    for page in payload.get('pages',[]):
        for row in build_rows(page):
            words=row['words']; text=' '.join(str(w['text']) for w in words)
            # Single explicit colon only; avoid mixing adjacent label/value pairs.
            m=re.fullmatch(r'\s*([^:]+?)\s*:\s*([^:]+)\s*',text)
            if not m: continue
            label,value=m.groups(); start=m.start(2); end=m.end(2); cursor=0; selected=[]
            for w in words:
                t=str(w['text']); stop=cursor+len(t)
                if cursor>=start and stop<=end: selected.append(w)
                cursor=stop+1
            if ' '.join(str(w['text']) for w in selected).strip()!=value.strip(): continue
            emit(page,label,value,words_box(selected),text)
        # Only unambiguous two-cell native table rows; no guessed column ownership.
        for table in page.get('tables') or []:
            if not isinstance(table,dict): continue
            for row in table.get('rows') or []:
                cells=row.get('cells',[]) if isinstance(row,dict) else row
                if not isinstance(cells,list) or len(cells)!=2 or not all(isinstance(c,dict) for c in cells): continue
                a,b=cells; label=str(a.get('text') or '').strip(); value=str(b.get('text') or '').strip()
                box=b.get('bbox')
                if box and len(box)==4:
                    selected=[w for w in page.get('words',[]) if box[0]<= (w['bbox'][0]+w['bbox'][2])/2 <=box[2] and box[1]<= (w['bbox'][1]+w['bbox'][3])/2 <=box[3]]
                    if norm(' '.join(str(w['text']) for w in selected))==norm(value): emit(page,label,value,words_box(selected),label+' | '+value)
    return out
