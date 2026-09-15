"""Render verified artifacts, learning curves and the final Turkish report."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from comparison.standardized.common_input_improvement import ROOT

results=json.loads((ROOT/'results_verified.json').read_text())
verification=json.loads((ROOT/'verification.json').read_text())
assert verification['status']=='verified' and len(results)==5
replays=[r for method in ('protgnn','gsat','graphcare') for r in json.loads((ROOT/f'checkpoint_replay_{method}.json').read_text())]
assert len(replays)==10
baseline=next(r for r in json.loads(Path('comparison/standardized/performance_review_20260913/baseline/results.json').read_text()) if r['view']=='concepts' and r['loss']=='none')['validation']['macro_f1']
all_rows=[]
fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
for ax,method in zip(axes,('protgnn','graphcare','gsat')):
    for row in results:
        if row['method']!=method: continue
        history=json.loads((ROOT/'trials'/method/row['variant']/'history.json').read_text())
        all_rows.extend({'method':method,'variant':row['variant'],**r} for r in history)
        ax.plot([r['epoch_one_based'] for r in history],[r['macro_f1'] for r in history],label=row['variant'])
    ax.axhline(baseline,color='gray',ls=':',label='common-concept LR')
    if method=='protgnn': ax.axvline(21,color='black',ls='--',alpha=.5,label='MCTS projection')
    if method=='gsat':
        ax.axvline(11,color='black',ls='--',alpha=.4)
        ax.axvline(21,color='black',ls='--',alpha=.4,label='r curriculum')
    ax.set_title(method); ax.set_xlabel('Completed epoch (1-based)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
axes[0].set_ylabel('Validation macro-F1 (n=7,448)')
fig.suptitle('Common-input source-snapshot diagnostic | star | seed 1234 | no test selection')
fig.tight_layout(); fig.savefig(ROOT/'validation_learning_curves.png',dpi=160); fig.savefig(ROOT/'validation_learning_curves.svg')
keys=sorted(set().union(*(r.keys() for r in all_rows)))
with (ROOT/'learning_curves.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(all_rows)
lines=['## Gerçek sonuçlar — beş koşul tamamlandı','',
       'Her satır aynı **7.448 validation hastası**, tek sabit seed 1234, star ve 30 tamamlanmış epoch. Her koşul **13.980 optimizer update**; test inference yok. Best checkpoint yalnız validation macro-F1 ile seçildi. Epoch sütunu 1-based; checkpoint içindeki epoch 0-based.','',
       '| Yöntem | Loss | Best epoch | Best val macro-F1 | Accuracy | Balanced accuracy | Epoch30 macro-F1 | Parametre |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for r in results:
    b=r['best_validation']; last=r['last_validation']
    lines.append(f"| {r['method']} | {r['variant']} | {b['epoch_one_based']} | {b['macro_f1']:.6f} | {b['accuracy']:.6f} | {b['balanced_acc']:.6f} | {last['macro_f1']:.6f} | {r['parameter_count']} |")
lines += ['', '| Loss karşılaştırması (sqrt − current) | Δ best macro-F1 | Δ best accuracy | Δ best balanced accuracy | Δ epoch30 macro-F1 |', '|---|---:|---:|---:|---:|']
for method in ('protgnn','graphcare'):
    a=next(r for r in results if r['method']==method and r['variant']=='current')
    b=next(r for r in results if r['method']==method and r['variant']=='sqrt_inverse')
    delta=[b['best_validation'][k]-a['best_validation'][k] for k in ('macro_f1','accuracy','balanced_acc')]
    delta.append(b['last_validation']['macro_f1']-a['last_validation']['macro_f1'])
    lines.append('| '+method+' | '+' | '.join(f'{v:+.6f}' for v in delta)+' |')
lines += ['', f'Ortak 192-concept, ağırlıksız lojistik baseline: validation macro-F1 **{baseline:.6f}**, n=7.448 (önceki korunmuş gerçek fit; yeni girdi matrisinin aynı concept bilgisi taşıdığı audit ile doğrulandı). Bu baseline da source-snapshot sınırlarını taşır.', '', '### Mekanizmalar ve checkpoint durumu', '']
for r in results:
    if r['method']=='protgnn':
        bestphase='projection öncesi' if r['best_validation']['epoch']<20 else 'projection sonrası'
        lines.append(f"- ProtGNN {r['variant']}: epoch20 (0-based) gerçek MCTS projection, 90 prototype × 10 train candidate; **{r['projection_replaced']}/90** prototype pozitif similarity ile değiştirildi. Seçilen checkpoint **{bestphase}**; son checkpoint projection sonrasıdır.")
lines += ['- GSAT r=0.9/0.8/0.7 curriculum dönemleri kaydedildi; information loss ve stochastic attention aktiftir.',
          '- Her `last.pt`: next_epoch=30, updates=13.980, yüklenebilir optimizer ve RNG. Her `best.pt`: seçilen gerçek epoch ve model state. Tamamlanma `state.json=completed` ile doğrulandı.',
          '- Learning curves: `learning_curves.csv`, `validation_learning_curves.png`, `validation_learning_curves.svg`; bütün 150 epoch prediction dosyası ve beş history korunuyor.', '',
          '### Yorumun sınırı', '',
          'Bu tek-seed validation tanı karşılaştırmasıdır; güven aralığı, multi-seed üstünlük, test kazanımı veya klinik kullanılabilirlik kanıtı değildir. Loss farkları ortak girdi ve sabit bütçe içinde yorumlanabilir. Önceki native-input sonuçlarla fark yalnız input etkisine bağlanamaz; girdi kodlaması ve eğitim bütçesi de farklıdır. Validation üzerinde epoch/loss seçimi validation iyimserliği taşır; test açılmadı. MPS/PyG ve CPU/GraphCare farklı Torch ortamları kullandı; sürümler `runtime_versions.json` içinde, süre/capacity eşitliği iddiası yok.', '',
          '### Doğrulama ve kurtarma', '',
          f"- **{verification['protected_files_unchanged']} korunmuş dosyanın** SHA-256 ve boyutu başlangıçla aynı: özgün 18 koşul, eski caches, kaynak/split, önceki review ve snapshot'a alınan dirty tracked dosyalar. `preservation_before.json`, `preservation_after.json`, `verification.json`.",
          '- Bütün **150 validation prediction** dosyasından confusion-count tabanlı bağımsız NumPy accuracy/recall/F1 ve top-k yeniden hesaplandı; shared metrics helper çağrılmadı, kaydedilen altı metrik ile eşleşti.',
          f"- **10 best/last checkpoint** yeniden yüklenip 7.448 validation hastasında tekrar forward yapıldı; argmax birebir eşleşti. Maksimum probability farkı **{max(r['max_probability_difference'] for r in replays):.9g}**. `checkpoint_replay_*.json`.",
          '- Gerçek GSAT checkpoint smoke epoch1/466 update noktasında durdu; Python/NumPy/Torch/MPS RNG ve optimizer ile epoch2'+'\u2019'+'den devam edip 30 epoch tamamladı. `events.jsonl` restored olayı ve önceki state kaydı mevcut.',
          '- İlk GraphCare zinciri legacy Torch `torch.mps` API eksikliğiyle **ilk checkpoint/ilk update öncesinde** code1 verdi; sqrt koşulu o zincirde hiç başlamadı. Aynı checkpoint regression legacy ortamda kırmızıydı; ardından legacy `weights_only` API farkı da aynı testte yakalandı. Capability detection düzeltildi ve her iki interpreter'+'\u2019'+'da test edildi. Başarısız deneme/log silinmedi: `failed_attempts/graphcare_current_precheckpoint/`, `graphcare_current_failed_precheckpoint.log`.',
          '- Compatibility migration, eski source hash'+'\u2019'+'lerini ters dönüşümle doğruladı; girdi/model/loss değişikliği yok. GSAT epoch1 checkpoint hash'+'\u2019'+'i migration öncesi/sonrası aynıydı. Eski/yeni code binding `compatibility_migration.json` içinde. GraphCare sıfır update'+'\u2019'+'ten açıkça yeniden başladı; bu optimizer resume diye sunulmuyor. Sonraki iki GraphCare koşulu gerçekten completed ve checkpoint replay ile doğrulandı.',
          '- `exit None` bildirimleri sonuç sayılmadı; OS PID/log/state/history/checkpoint ayrı okundu. Tamamlanmış koşulun üstüne yeni eğitim yazılmadı.',
          '- Test çıktıları: `focused_tests_final.log`, `graphcare_tests_final.log`, `maintained_tests_final.log`. Test-first dilimler: feature/hub contract, train-fit, checkpoint/RNG restore, attempt binding ve warm-up/projection schedule. Gerçek singleton hub-only ve mixed-batch forward/backward her yöntemde test edildi. Tek-seferlik audit/runner/verifier için gerçek veri yürütmesi yapıldı; her evidence-script satırının TDD ile yazıldığı iddia edilmiyor.',
          '- Commit/push yapılmadı, production runner koruması ve eski caches değiştirilmedi. Tek seferlik yeni input materyalizasyonu yalnız izole köke yazıldı.', '']
report=Path('docs/common-input-improvement.md'); text=report.read_text()
start=text.index('## Sonuç durumu'); end=text.index('## Komutlar ve artefact',start)
text=text[:start]+'\n'.join(lines)+'\n'+text[end:]
report.write_text(text)
print('Verified result table and 150-point learning curves rendered.')
