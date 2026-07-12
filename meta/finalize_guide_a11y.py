from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
import tempfile, shutil

DOCX=Path('/mnt/data/GoalOS_Partner_MASTERCLASS_MASTERPIECE_v3/guides/GoalOS_Partner_MASTERCLASS_MASTERPIECE_Operating_Guide.docx')
ALTS=[
'GoalOS Partner MASTERCLASS APEX boardroom interface showing four presentation velocities for proof-carrying intelligence.',
'GoalOS Partner MASTERCLASS Masterpiece launcher with coordinated APEX, Grand, Sovereign and Production Showcase experiences.',
'Production Partner Showcase overview introducing GoalOS as the Proof OS for autonomous AI work.',
'Canonical GoalOS proof-to-capability loop from Objective through proof, Chronicle, validated capability and future mission improvement.',
'AGI Node Council interface showing seven complementary reviewer perspectives and committee synthesis.',
'Validated Skill Graph diagram showing a scoped capability passport with evidence, tests, validators, replay, risk and rollback.',
'AEP-001 public-private proof boundary separating controlled private intelligence from minimum public proof commitments.',
'Governed recursive self-improvement kernel showing TARGET, EMIT, FILTER, ATLAS, TEST-PLAN, EVAL, INSERT and PROMOTE.',
'Move-37 breakthrough handling interface requiring reproduction, stress testing, persistence and dossier packaging.',
'Mission 1 to Mission 2 transfer interface comparing fresh control, raw memory, validated skill and ungated candidate under equal constraints.',
'GoalOS Deal Architect interface converting partner interest into a bounded evidence-producing engagement.'
]
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
W=NS['w']

def patch_xml(data: bytes, part: str) -> bytes:
    root=etree.fromstring(data)
    if part=='word/document.xml':
        docprs=root.xpath('.//wp:docPr',namespaces=NS)
        if len(docprs)!=len(ALTS):
            raise RuntimeError(f'Expected {len(ALTS)} images, found {len(docprs)}')
        for el,alt in zip(docprs,ALTS):
            el.set('descr',alt)
            el.set('title',alt.split('.')[0])
    for tbl in root.xpath('.//w:tbl',namespaces=NS):
        rows=tbl.xpath('./w:tr',namespaces=NS)
        if not rows: continue
        tr=rows[0]
        trPr=tr.find(f'{{{W}}}trPr')
        if trPr is None:
            trPr=etree.Element(f'{{{W}}}trPr')
            tr.insert(0,trPr)
        hdr=trPr.find(f'{{{W}}}tblHeader')
        if hdr is None:
            hdr=etree.SubElement(trPr,f'{{{W}}}tblHeader')
        hdr.set(f'{{{W}}}val','true')
    return etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone='yes')

with tempfile.TemporaryDirectory() as td:
    out=Path(td)/DOCX.name
    with ZipFile(DOCX,'r') as zin, ZipFile(out,'w',ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data=zin.read(item.filename)
            if item.filename in ('word/document.xml','word/footer1.xml'):
                data=patch_xml(data,item.filename)
            zout.writestr(item,data)
    shutil.copy2(out,DOCX)
print(DOCX)
