import os
import json
import importlib.util
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from core.db_manager import DBManager


def load_review_module():
    path = ROOT / 'src/app/pages/2_Review.py'
    spec = importlib.util.spec_from_file_location('review', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_save_correction_updates_db_and_template(tmp_path):
    os.chdir(tmp_path)
    # setup workspace and files
    workspace_doc = tmp_path / 'workspace' / 'DOC_1'
    crops = workspace_doc / 'crops'
    crops.mkdir(parents=True, exist_ok=True)
    extract = workspace_doc / 'extract.json'
    data = {'field': {'text': 'OLD', 'needs_human': True, 'source_image': 'P1_field.png', 'result_id': 1}}
    with open(extract, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    with open(workspace_doc / 'template.json', 'w', encoding='utf-8') as f:
        json.dump({'name': 'invoice'}, f, ensure_ascii=False)

    # templates
    templates_dir = tmp_path / 'templates'
    templates_dir.mkdir()
    with open(templates_dir / 'invoice.json', 'w', encoding='utf-8') as f:
        json.dump({'name': 'invoice', 'rois': {}, 'corrections': []}, f, ensure_ascii=False)

    # database
    db_dir = tmp_path / 'database'
    db_dir.mkdir()
    db = DBManager(str(db_dir / 'ocr_results.db'))
    db.initialize()
    job_id = db.create_job('invoice', '2025-01-01T00:00:00')
    db.add_result(job_id, 'img.png', 'field', 'invoice', final_text='OLD')
    db.close()

    item = {
        'doc': 'DOC_1',
        'key': 'field',
        'text': 'OLD',
        'source': 'P1_field.png',
        'extract_path': str(extract),
        'crops_dir': str(crops),
        'data': data,
        'result_id': 1,
    }

    review = load_review_module()
    review.save_correction(item, 'NEW', add_dict=True)

    # extract.json updated
    with open(extract, 'r', encoding='utf-8') as f:
        updated = json.load(f)
    assert updated['field']['text'] == 'NEW'
    assert 'needs_human' not in updated['field']

    # DB updated
    db2 = DBManager(str(db_dir / 'ocr_results.db'))
    results = list(db2.fetch_results(job_id))
    db2.close()
    assert results[0]['final_text'] == 'NEW'
    assert results[0]['corrected_by_user'] == 1
    assert results[0]['status'] == 'confirmed'

    # template corrections updated
    with open(templates_dir / 'invoice.json', 'r', encoding='utf-8') as f:
        tpl = json.load(f)
    assert {"wrong": "OLD", "correct": "NEW"} in tpl['corrections']
