"""High precision schedule evidence, independent of insurer-specific values."""
import re
from field_extractor import DATE_RE, INSURERS, _value_from
from field_validators import valid_date, valid_policy_number
from spatial_extractor import build_rows, find_value_bbox, words_text

def extract_schedule(page, kind):
    rows=build_rows(page)
    def emit(field,value,words,evidence,score=200):
        box=find_value_bbox(words,value)
        if box and value: return (field,value,box,evidence,score)
    def span(row,match,group=1):
        a,b=match.span(group); pos=0; out=[]
        for w in row['words']:
            end=pos+len(w['text'])
            if pos<b and end>a: out.append(w)
            pos=end+1
        return out
    def item(field,row,m,group=1,score=200):
        value=m.group(group).strip(); return emit(field,value,span(row,m,group),row['text'],score)
    for i,row in enumerate(rows):
        t=row['text']; low=t.lower()
        # Explicit current policy labels, never previous policy or receipt.
        m=re.match(r'^\s*(?:master\s+)?policy\s*(?:number|no\.?)\s*[:.-]?\s*([A-Z0-9/-]+(?:\s+\d{3,4}){0,6})',t,re.I)
        if m:
            v=m.group(1); compact=re.sub(r'\s+','',v)
            if valid_policy_number(compact):
                x=emit('Policy Number',compact,span(row,m),t,230 if '/' not in compact else 205)
                if x: yield x
        # Bound date search to the policy-period region, not issue/previous dates.
        if re.match(r'^\s*(?:period of insurance|policy period|period of cover|insurance period)\b',t,re.I):
            region=' '.join(r['text'] for r in rows[i:i+3]); region=re.split(r'\b(?:policy issuance date|policy term|date of proposal|previous policy|prev\. policy)\b',region,flags=re.I)[0]
            ds=[m.group() for m in DATE_RE.finditer(region) if valid_date(m.group())]
            if len(ds)>=2:
                ws=[w for r in rows[i:i+3] for w in r['words']]
                for f,v in zip(('Policy Start','Policy End'),ds[:2]):
                    x=emit(f,v,ws,region,230)
                    if x:yield x
        # Insurer can appear in a signature as well as header.
        for name in INSURERS:
            x=emit('Insurer',name,row['words'],t,220)
            if x:yield x
        # Explicit customer name, stop at unrelated columns.
        m=re.match(r'^(?:insured name|customer name|proposer name)\s*[:.-]?\s*(.+?)(?=\s+(?:RTO|SAC Code|Issuing Office|Office Code|GSTIN)\b|$)',t,re.I)
        if m:
            f = 'Proposer Name' if low.startswith('proposer') else 'Insured Name' if low.startswith('insured') else 'Customer Name'
            v=_value_from(m.group(1),f)
            if v and v.lower()!='code':
                x=emit(f,v,span(row,m),t,220)
                if x:yield x
        for f, pattern in [('Family Floater Cover',r'\bBasic Floater Sum Insured\s*:\s*Rs\.?\s*([\d,]+(?:\.\d+)?)'),('Sum Insured / Assured',r'\bLimit of Coverage\s*:\s*Rs\.?\s*([\d,]+(?:\.\d+)?)')]:
            m=re.search(pattern,t,re.I)
            if m:
                x=item(f,row,m,score=225)
                if x:yield x
        # Exact amount labels, excluding percentage and prose.
        m=re.match(r'^(?:total premium|total sum insured|total sum assured|basic floater sum insured|limit of coverage)\s*[: ]*\s*(?:Rs\.?|INR|₹|`)?\s*([\d,]+(?:\.\d{1,2})?)(?:/-)?\s*$',t,re.I)
        if m:
            f='Total Premium' if low.startswith('total premium') else 'Family Floater Cover' if low.startswith('basic floater') else 'Sum Insured / Assured'
            x=item(f,row,m,score=225)
            if x:yield x
        # Tax tables have rate followed by amount, never use rate as amount.
        m=re.match(r'^(CGST|SGST|IGST)\s+(\d+(?:\.\d+)?)\s+([\d,]+(?:\.\d+)?)$',t,re.I)
        if m:
            x=item(m.group(1).upper(),row,m,3,230)
            if x:yield x
        # Premium / GST / Total header, with possible receipt column.
        if re.match(r'^Premium\s+GST\s+Total\b',t,re.I) and i+1<len(rows):
            data=rows[i+1]; m=re.match(r'^\s*[`₹]?\s*([\d,]+(?:\.\d+)?)\s+[`₹]?\s*([\d,]+(?:\.\d+)?)\s+[`₹]?\s*([\d,]+(?:\.\d+)?)',data['text'])
            if m:
                for j,f in enumerate(('Premium','GST','Total Premium'),1):
                    x=item(f,data,m,j,235)
                    if x:yield x
        # Vehicle header variant must be treated as a single column.
        if kind=='Vehicle Insurance' and re.match(r'^Make\s+Model\s*-\s*Variant\s+Engine No\s+Chassis No\s+Mfg Yr\s+Seats\s+Body Type\s+CC$',t,re.I) and i+1<len(rows):
            data=rows[i+1]; m=re.match(r'^(\S+)\s+(.+?)\s+([A-Z0-9/-]{5,})\s+([A-Z0-9/-]{6,})\s+(\d{4})\s+(\d+)\s+(.+?)\s+(\d+)$',data['text'],re.I)
            if m:
                for j,f in enumerate(('Vehicle Make','Vehicle Model','Engine Number','Chassis Number','Manufacturing Year','Seating Capacity','Body Type','CC'),1):
                    x=item(f,data,m,j,235)
                    if x:yield x
        # Pincode only in the explicitly insured name/address block.
        if re.match(r'^Name\s*(?:&|and)\s*Address of Insured\b',t,re.I):
            for r in rows[i:i+3]:
                m=re.search(r'(?<!\d)([1-9]\d{5})(?!\d)',r['text'])
                if m:
                    x=item('Pincode',r,m,score=225)
                    if x:yield x
        # Total sum assured may be printed on the following line.
        if re.search(r'\btotal sum assured of Rs\.?\s*$',t,re.I) and i+1<len(rows):
            data=rows[i+1]; m=re.match(r'^([\d,]+(?:\.\d+)?)$',data['text'])
            if m:
                x=item('Sum Insured / Assured',data,m,score=225)
                if x:yield x

# Section-owned columns: derive the boundary from printed section headings
# and the paired label separators, rather than using a fixed page percentage.
def extract_owned_contacts(page):
    from spatial_extractor import words_box
    rows=build_rows(page)
    for i,row in enumerate(rows):
        if not re.search(r"insured(?:'s)? details",row['text'],re.I) or not re.search(r'issuing office details',row['text'],re.I):
            continue
        block=[]
        for r in rows[i+1:]:
            if re.search(r'^policy details\b',r['text'],re.I): break
            block.append(r)
            if len(block)>=25: break
        paired=next((r for r in block if sum(w['text']==':' for w in r['words'])==2),None)
        if not paired: continue
        colons=[w['bbox'][0] for w in paired['words'] if w['text']==':']
        right_label=next((w['bbox'][0] for w in paired['words'] if w['bbox'][0]>colons[0] and w['text'].lower()=='office'),None)
        if right_label is None: continue
        for side,(lo,hi,sep) in enumerate([(0,right_label-2,colons[0]),(right_label-2,float('inf'),colons[1])]):
            entries=[]
            for r in block:
                ws=[w for w in r['words'] if lo<=w['bbox'][0]<hi]
                labels=[w for w in ws if w['bbox'][0]<sep-3]
                vals=[w for w in ws if w['bbox'][0]>sep+3]
                label=words_text(labels).lower().strip(' :')
                if label: entries.append([label,vals])
                elif entries and vals: entries[-1][1].extend(vals)
            mapping=({'address':'Residential Address','phone no':'Phone Number','e-mail/fax':'Email ID','pan no':'Insured PAN','gstin/uin':'Insured GSTIN','customer id':'Customer ID'} if side==0 else {'address':'Issue Office Address','phone no':'Office Phone','e-mail/fax':'Office Email','gstin':'Issue Office GSTIN','office code':'Issue Office Name','sac':'SAC Code'})
            for label,ws in entries:
                f=mapping.get(label.rstrip('.'))
                if not f or not ws: continue
                raw=words_text(ws)
                pattern={'Email ID':r'[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}','Office Email':r'[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}','Insured GSTIN':r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b','Issue Office GSTIN':r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b','SAC Code':r'\b\d{6}\b'}.get(f)
                m=re.search(pattern,raw) if pattern else None
                value=m.group() if m else raw if not pattern else ''
                if not value: continue
                box=find_value_bbox(ws,value) or words_box(ws)
                if box: yield f,value,box,label+': '+raw,300
                if f=='Residential Address':
                    m=re.search(r'\b[1-9]\d{5}\b',raw)
                    if m:
                        b=find_value_bbox(ws,m.group())
                        if b: yield 'Pincode',m.group(),b,raw,300
                if f=='Issue Office Name':
                    m=re.search(r'\((\d{4,})\)',raw)
                    if m:
                        b=find_value_bbox(ws,m.group(1))
                        if b: yield 'Issue Office Code',m.group(1),b,raw,300
