"""Run: python regression_test.py PATH_TO_DATASET_ZIP (no model download)."""
import sys,zipfile,tempfile
from pathlib import Path
from layoutlmv3_pipeline import process_pdf
from unified_extractor import extract_fields
EXPECTED={
'018':{'Policy Number':'2319100094851200000','Policy Start':'24/12/2015','Policy End':'23/12/2016','Vehicle Model':'SWIFT - ZXI','Total Premium':'1944','CC':'1298'},
'003':{'Policy Number':'73010034220400000001','Policy Start':'01/04/2022','Policy End':'31/03/2023','GST':'432606','IGST':'432606','Total Premium':'2835970'},
'011':{'Policy Start':'22/01/2026','Renewal Date':'22/01/2027','Sum Insured / Assured':'20,66,35,000.00','Number of Members':'3179'},
'054':{'Policy Number':'FM00111573000101','Policy Start':'25/01/2025','Policy End':'24/01/2026','Total Premium':'5,55,099','Sum Insured / Assured':'1,00,09,00,000','Pincode':'626123'},
'010':{'Policy Number':'4952112006019567','Policyholder':'NAGESH R','Policy Start':'27-Oct-2025','Policy End':'26-Oct-2026','Total Premium':'20,927','Family Floater Cover':'5,00,000'}}
def main():
 passed=total=0
 with zipfile.ZipFile(sys.argv[1]) as z, tempfile.TemporaryDirectory() as td:
  for prefix,expected in EXPECTED.items():
   name=next(n for n in z.namelist() if Path(n).name.startswith(prefix) and n.lower().endswith('.pdf'))
   path=Path(td)/Path(name).name;path.write_bytes(z.read(name))
   payload,_=process_pdf(path,Path(td)/'results',None,None,'cpu',max_pages=3)
   fields={f['field']:f for f in extract_fields(payload)}
   for field,value in expected.items():
    actual=fields.get(field,{}); ok=actual.get('value')==value and actual.get('bbox') is not None
    total+=1;passed+=ok
    print(prefix,field,'PASS' if ok else 'FAIL',actual.get('value'),actual.get('status'))
 print(f'{passed}/{total} value + source-box checks passed')
 return passed!=total
if __name__=='__main__':sys.exit(main())
