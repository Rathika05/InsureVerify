"""Evidence-grounded spatial extraction for schedule tables and sections.
Works with layoutlmv3_pipeline payload: normalized 0..1000 word boxes.
No Qwen or semantic claims about base LayoutLMv3.
"""
import re

DATE = re.compile(r'(?<!\d)(?:\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{2}-\d{2})(?!\d)')
AMOUNT = re.compile(r'(?<![\w/])(?:[₹`]|Rs\.?\s*|INR\s*)?\s*(\d[\d,]*(?:\.\d{1,2})?)(?!\w)', re.I)

def clean(t):
    return re.sub(r'\s+', ' ', str(t)).strip(' :|\t')

def norm(t):
    return re.sub(r'[^a-z0-9]', '', str(t).lower())

def bbox(ws):
    return [min(w['bbox'][0] for w in ws), min(w['bbox'][1] for w in ws),
            max(w['bbox'][2] for w in ws), max(w['bbox'][3] for w in ws)] if ws else None

def locate(words, value, region=None):
    """Return exact token span; region restricts candidate to a table cell."""
    target = norm(value)
    if not target: return None
    candidates = [w for w in words if not region or
                  (region[0]-4 <= (w['bbox'][0]+w['bbox'][2])/2 <= region[2]+4 and
                   region[1]-4 <= (w['bbox'][1]+w['bbox'][3])/2 <= region[3]+4)]
    for i in range(len(candidates)):
        joined = ''
        for j in range(i, min(i+16, len(candidates))):
            joined += norm(candidates[j]['text'])
            if joined == target: return bbox(candidates[i:j+1])
            if len(joined) > len(target): break
    return None

def lines(page):
    """Native line grouping by PDF block/line, retaining spatial coordinates."""
    groups = {}
    for i,w in enumerate(page.get('words', [])):
        groups.setdefault((w.get('block',0), w.get('line',0)), []).append((i,w))
    out=[]
    for seq in groups.values():
        seq.sort(key=lambda x:x[1]['bbox'][0])
        ws=[w for _,w in seq]
        out.append({'text':clean(' '.join(w['text'] for w in ws)), 'box':bbox(ws), 'words':ws,
                    'first_index':min(i for i,_ in seq)})
    return sorted(out,key=lambda x:(x['box'][1],x['box'][0]))

def field(name,value,page,box,evidence):
    return {'field':name,'value':clean(value),'page':page['page'],'bbox':box,
            'status':'Found' if box else 'Needs review','evidence':clean(evidence)[:250]}

def candidate(page,name,value,evidence,region=None):
    if not value:return None
    return field(name,value,page,locate(page['words'],value,region),evidence)

def period(page):
    """Read multiline Period of Insurance / Policy period, avoiding proposal dates."""
    ls=lines(page)
    for i,l in enumerate(ls):
        if not re.search(r'\b(?:period of insurance|policy period)\b',l['text'],re.I):continue
        # Prefer explicit From and To; limit to 3 nearby lines, not unrelated table rows.
        context=' '.join(x['text'] for x in ls[i:i+3] if x['box'][1]-l['box'][1] < 80)
        dates=list(DATE.finditer(context))
        if len(dates)<2:continue
        start,end=dates[0].group(),dates[1].group()
        if start==end:continue
        return [candidate(page,'Policy Start',start,context),candidate(page,'Policy End',end,context)]
    # text-order fallback for native PDF table where columns split into different blocks
    text=page.get('text','')
    m=re.search(r'(?:period of insurance|policy period)\s*:\s*from\s*:?\s*('+DATE.pattern+r')[^\n]{0,90}(?:\n\s*)?('+DATE.pattern+r')',text,re.I)
    if m:
        return [candidate(page,'Policy Start',m.group(1),m.group(0)),candidate(page,'Policy End',m.group(2),m.group(0))]
    return []

def premium_table(page):
    """Associate numeric cells with Premium/GST/Total header using X alignment.
    Also supports premium-and-GST-details section on page 2.
    """
    ls=lines(page)
    for i,head in enumerate(ls):
        if not re.fullmatch(r'premium',head['text'],re.I):continue
        y=head['box'][1]
        # Nearby headers at similar vertical position (columns).
        near=[x for x in ls if abs(x['box'][1]-y)<17]
        gst=next((x for x in near if re.fullmatch(r'(?:gst|igst)',x['text'],re.I)),None)
        total=next((x for x in near if re.fullmatch(r'total',x['text'],re.I)),None)
        if not gst or not total:continue
        headers={'Premium':head,'GST':gst,'Total Premium':total}
        output=[]
        for label,h in headers.items():
            cx=(h['box'][0]+h['box'][2])/2
            # Look for the first amount below the header, close to the column center.
            matches=[]
            for row in ls:
                dy=row['box'][1]-h['box'][3]
                if not 0<=dy<=35:continue
                rx=(row['box'][0]+row['box'][2])/2
                if abs(rx-cx)>95:continue
                text=row['text']
                # Full cell should be a currency amount, not a receipt number or words.
                m=re.fullmatch(r'\s*[₹`]?\s*(\d[\d,]*(?:\.\d{1,2})?)\s*',text)
                if not m:continue
                if not 100<=float(m.group(1).replace(',',''))<1e11:continue
                matches.append((abs(rx-cx)+dy/3,m.group(1),row))
            if matches:
                _,val,row=min(matches,key=lambda x:x[0])
                output.append(candidate(page,label,val,f'{label}: {val}',row['box']))
        if len(output)>=2:return output
    # Section table with 'Premium' at left and '₹ 2403364.00' far right.
    for i,l in enumerate(ls):
        if not re.fullmatch(r'premium',l['text'],re.I):continue
        for v in ls:
            if abs(v['box'][1]-l['box'][1])>12 or v['box'][0]<=l['box'][2]:continue
            m=re.fullmatch(r'\s*[₹`]?\s*(\d[\d,]*(?:\.\d{1,2})?)\s*',v['text'])
            if m and float(m.group(1).replace(',',''))>=100:
                return [candidate(page,'Premium',m.group(1),f'Premium: {m.group(1)}',v['box'])]
    return []

def tpa(page):
    ls=lines(page)
    for i,l in enumerate(ls):
        if not re.search(r'\bdetails of tpa\b',l['text'],re.I):continue
        bottom=ls[min(i+12,len(ls)-1)]['box'][3]
        for j in range(i+1,min(i+12,len(ls))):
            row=ls[j]
            if row['box'][1]-l['box'][1]>110:break
            if re.fullmatch(r'name\s*:',row['text'],re.I) or re.fullmatch(r'name',row['text'],re.I):
                # find value to the right of Name on the same row
                opts=[v for v in ls if v['box'][0]>row['box'][2] and
                      abs(v['box'][1]-row['box'][1])<12 and v['box'][0]<650]
                opts=[v for v in opts if re.search(r'[a-z]{2}',v['text'],re.I) and
                      not re.match(r'^(?:telephone|address|fax|email)\b',v['text'],re.I)]
                if opts:
                    v=min(opts,key=lambda x:x['box'][0])
                    return candidate(page,'Hospital / TPA',v['text'],f'Details of TPA | Name: {v["text"]}',v['box'])
            m=re.match(r'^Name\s*:\s*(.{4,80})$',row['text'],re.I)
            if m:
                val=m.group(1).split('Telephone')[0].strip()
                return candidate(page,'Hospital / TPA',val,f'Details of TPA | {row["text"]}')
    return None

def conditions(page):
    """Extract separate waiting-period choices, not a misleading generic number."""
    result=[]
    ls=lines(page)
    mapping={
        '9 Months Waiting Period Deleted':r'deletion of 9 months waiting period',
        '30 Days Waiting Period Deleted':r'deletion of 30 days waiting period',
        '2/4 Year Exclusion Deleted':r'deletion of 2/4 year exclusion',
        'Pre-existing Cover Opted':r'pre-existing cover opted',
    }
    for name,pat in mapping.items():
        for l in ls:
            if not re.search(pat,l['text'],re.I):continue
            m=re.search(r'\b(YES|NO)\b\s*$',l['text'],re.I)
            if m:
                result.append(candidate(page,name,m.group(1).upper(),l['text']))
                break
            neighbors=[x for x in ls if x['box'][0]>l['box'][2] and abs(x['box'][1]-l['box'][1])<12]
            yesno=next((x for x in neighbors if x['text'].upper() in ('YES','NO')),None)
            if yesno:
                result.append(candidate(page,name,yesno['text'].upper(),l['text']+' | '+yesno['text'],yesno['box']))
                break
    return result

def enhance_fields(payload, base_fields, kind):
    """Drop-in enhancement: only replace fields with spatially grounded evidence.
    Returns same dict contract used by Streamlit app.
    """
    by_name={f['field']:dict(f) for f in base_fields}
    extra=[]
    for page in payload.get('pages',[]):
        found=period(page)+premium_table(page)
        if kind in ('Health Insurance','Family Insurance'):
            x=tpa(page)
            if x:found.append(x)
            found+=conditions(page)
        for f in found:
            if not f:continue
            name=f['field']
            if name in by_name:
                old=by_name[name]
                if old['status']!='Found' and f['status']=='Found':by_name[name]=f
                elif old['status']=='Not found':by_name[name]=f
            else:
                # Avoid appending multiple copies of same special field.
                if not any(x['field']==name for x in extra):extra.append(f)
    return list(by_name.values())+extra
