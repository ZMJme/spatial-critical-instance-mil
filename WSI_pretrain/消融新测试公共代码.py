import os
os.environ['OMP_NUM_THREADS'] = '1'
import csv
import gc
import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 修改这三个路径。没有训练代码，不重新定义模型类。
MODEL_ROOT = Path('/root/autodl-tmp/code/WSI/WSI_pretrain/model/MIL/Ablation_NEW/CAMELYON16')
TEST_ROOT = Path('/root/autodl-tmp/database/CAMELYON16/pkl_data/testing')
SAVE_ROOT = Path('/root/autodl-tmp/新测试结果/消融实验')


def run_ablation_test(experiment, reader_name, scale='20xbags', required=()):
    missing = [name for name in [reader_name, *required] if name not in globals()]
    if missing:
        raise NameError(f'请先恢复原Notebook中的定义（不要执行训练）：{missing}')
    reader = globals()[reader_name]
    run_dir = MODEL_ROOT / experiment
    weight = run_dir / 'model.pth'
    for filename in ['model.pth', 'config.json', 'training_summary.json']:
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(run_dir / filename)
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    if config.get('experiment') != experiment or config.get('class_num') != 2:
        raise ValueError('配置与实验不一致，请检查权重目录')
    test_dir = TEST_ROOT / scale
    files = sorted(test_dir.glob('test_*.pth'))
    by_id = {}
    for path in files:
        sid = re.match(r'test_\d+', path.name).group()
        if sid in by_id:
            raise ValueError(f'重复切片：{sid}')
        by_id[sid] = path
    if len(by_id) != 129:
        raise ValueError(f'{scale}预期129张，实际{len(by_id)}张')

    # C1/C2保留原保存的测试名单顺序；按切片ID匹配修正后的文件名。
    split_path = run_dir / 'split_on_CAMELYON16.json'
    if scale != '20xbags':
        split = json.loads(split_path.read_text(encoding='utf-8'))
        saved_ids = [re.match(r'test_\d+', Path(name).name).group() for name in split['test']]
        if len(saved_ids) != len(set(saved_ids)) or set(saved_ids) != set(by_id):
            raise ValueError('当前测试切片与原保存名单不一致')
        files = [by_id[sid] for sid in saved_ids]

    # 先检查全部缓存内部标签，再开始模型推理；不读取reference.csv。
    labels_before = {}
    for sid, path in by_id.items():
        data = torch.load(path, map_location='cpu', weights_only=False)
        labels_before[sid] = int(data['label'])
        del data
    if labels_before.get('test_114') != 1:
        raise ValueError(f'{scale}的test_114内部label仍非1；仅改文件名不够')
    if Counter(labels_before.values()) != {0: 80, 1: 49}:
        raise ValueError(f'缓存标签应为80阴性/49阳性：{dict(Counter(labels_before.values()))}')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = SAVE_ROOT / f'{experiment}_新测试_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f'ablation_retest_{experiment}_{stamp}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(output / 'log.log', mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = None
    try:
        logger.info('新测试，不训练；权重=%s；读取函数=%s', weight.resolve(), reader_name)
        logger.info('原配置=%s', json.dumps(config, ensure_ascii=False))
        (output / '原检查点_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        # 与原Notebook一致：加载完整模型。仅加载自己生成的可信检查点。
        model = torch.load(weight, map_location=device, weights_only=False)
        if not isinstance(model, nn.Module):
            raise TypeError('此代码对应原Notebook保存的完整模型，不接受其他格式权重')
        model.to(device).eval()
        logger.info('实际模型结构：\n%s', model)
        rows = []
        with torch.no_grad():
            for path in files:
                features, edges, target = reader(path, device)
                sid = re.match(r'test_\d+', path.name).group()
                label = int(target.item())
                if label != labels_before[sid]:
                    raise ValueError(f'{sid}读取函数返回的标签与缓存不一致')
                if experiment.startswith('A1_'):
                    nodes = torch.arange(features.shape[0], device=device, dtype=torch.long)
                    if not torch.equal(edges, torch.stack([nodes, nodes])):
                        raise ValueError('A1必须使用原仅自环边')
                _, logits, _, _ = model(features, edges)
                if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
                    raise ValueError(f'{path.name}输出形状或数值异常')
                pred = int(logits.argmax(dim=1).item())
                prob = float(logits.softmax(dim=1)[0, 1].item())
                rows.append(dict(file=path.name, slide_id=sid, label=label, prediction=pred,
                                 tumor_probability=prob, logits=logits[0].cpu().tolist(),
                                 patch_count=int(features.shape[0])))
                logger.info('%s | label=%d | prediction=%d | probability=%.8f', sid, label, pred, prob)
        y = [r['label'] for r in rows]
        p = [r['prediction'] for r in rows]
        cm = confusion_matrix(y, p, labels=[0, 1])
        tn, fp, fn, tp = map(int, cm.ravel())
        result = dict(model=experiment, run_type='新测试', n=len(rows), negative=80, positive=49,
                      precision=float(precision_score(y, p, zero_division=0)),
                      accuracy=float(accuracy_score(y, p)), recall=float(recall_score(y, p, zero_division=0)),
                      f1=float(f1_score(y, p, zero_division=0)), TN=tn, FP=fp, FN=fn, TP=tp,
                      confusion_matrix_labels_0_1=cm.tolist(), weight_path=str(weight.resolve()),
                      test_data_path=str(test_dir.resolve()))
        for suffix, content in [('metrics', result), ('predictions', rows)]:
            name = f'{experiment}_新测试_{suffix}'
            (output / f'{name}.json').write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
            records = content if isinstance(content, list) else [content]
            with (output / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        info = dict(checkpoint_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
                    reader=reader_name, scale=scale, torch_version=torch.__version__, device=str(device),
                    prediction_rule='bag logits argmax', label_source='cache internal label', training=False)
        (output / '新测试_run_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        if split_path.is_file():
            (output / '原split_on_CAMELYON16.json').write_bytes(split_path.read_bytes())
        summary = (f"TEST {experiment} 新测试 | Pre: {result['precision']:.6f} | "
                   f"Acc: {result['accuracy']:.6f} | Rec: {result['recall']:.6f} | F1: {result['f1']:.6f}")
        logger.info(summary)
        logger.info('混淆矩阵[0,1]=%s', cm.tolist())
        print(summary)
        print(f'结果及log.log：{output}')
    except Exception:
        logger.exception('新测试失败')
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
