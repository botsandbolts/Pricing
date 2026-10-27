"""Refresh frozen Kaggle source archives after training-code changes."""
import base64,hashlib,io,json,re,zipfile
from pathlib import Path


def refresh():
    root=Path(__file__).resolve().parents[1]
    for notebook_file in root.glob('train*_kaggle.ipynb'):
        notebook=json.loads(notebook_file.read_text())
        for cell in notebook['cells']:
            source=''.join(cell['source'])
            match=re.search(r"SOURCE_ZIP\s*=\s*'([^']+)'",source)
            if not match:continue
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(match.group(1)))) as z:
                contents={n:(root/n).read_bytes() if (root/n).is_file() else z.read(n) for n in z.namelist()}
            for package in ['pricing','carbids','marketbridge','scripts']:
                for file in (root/package).glob('*.py'):
                    contents[str(file.relative_to(root))]=file.read_bytes()
            for name in ['requirements-cleanlab.txt','CLEANLAB.md','data/description-examples.json']:
                contents[name]=(root/name).read_bytes()
            output=io.BytesIO()
            with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
                for name in sorted(contents):
                    entry=zipfile.ZipInfo(name,date_time=(2026,10,27,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
                    z.writestr(entry,contents[name])
            payload=output.getvalue();encoded=base64.b64encode(payload).decode()
            source=source[:match.start(1)]+encoded+source[match.end(1):]
            source=re.sub(r"(SOURCE_SHA(?:256)?\s*=\s*)'[0-9a-f]{64}'",lambda m:m.group(1)+repr(hashlib.sha256(payload).hexdigest()),source)
            cell['source']=source.splitlines(keepends=True)
        notebook_file.write_text(json.dumps(notebook,indent=1)+'\n')

if __name__=='__main__':refresh()
