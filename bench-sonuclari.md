# Ölçüm Sonuçları — Recall@k

**Tarih:** 9 Ağustos 2026
**Üretim:** `python tests/bench.py --dataset <klasör> --tiles 3 --out bench.md`

---

## Veri seti ve önemli uyarı

Aşağıdaki sayılar `tests/make_dataset.py` ile üretilen **sentetik** bir set
üzerindedir: 12 desen ailesi × 5 senaryo = 60 görsel. Her aile için orijinal,
farklı renk (colorway), 90° döndürülmüş, yeniden sıkıştırılmış kopya ve
kırpılmış (BMP) varyantlar üretilir.

> **Bu sayılar gerçek arşiv performansını temsil etmez.** Set iki yönden
> zorlayıcıdır: (1) tüm desenler aynı geometrik üreteçten gelir, dolayısıyla
> "alakasız" aileler gerçekte olacağından benzerdir; (2) paletler aileler
> arasında dönüşümlü kullanılır, yani bir ailenin colorway varyantı başka bir
> ailenin orijinaliyle **aynı renktedir** — bu, renge duyarlı modelleri
> (DINOv2) sistematik olarak cezalandırır.
>
> Karar vermeden önce ölçümü kendi arşivinizden derlenmiş 300–500 görselle
> tekrarlayın. Dosya adlandırması `<aile>_<senaryo>.<uzantı>` olmalıdır
> (ör. `desen042_recolor.jpg`); harness aileyi ilk alt çizgiye kadar,
> senaryoyu sonrasından okur.

---

## Sonuçlar

| Arka uç | TTA | hit@1 | hit@5 | hit@10 | recall@5 | recall@10 | sorgu(ms) |
|---|---|---|---|---|---|---|---|
| hash | kapalı | 0.35 | 0.53 | 0.60 | 0.24 | 0.28 | 2.3 |
| hash | açık | 0.55 | 0.77 | 0.78 | 0.45 | 0.53 | 8.9 |
| embedding | kapalı | 0.43 | 0.60 | 0.73 | 0.29 | 0.47 | 115.4 |
| embedding | açık | 0.47 | 0.60 | 0.67 | 0.28 | 0.45 | 777.8 |
| embedding + 3×3 karo | açık | 0.48 | 0.67 | 0.85 | 0.30 | 0.53 | 761.7 |
| **hibrit + 3×3 karo** | açık | **0.68** | **0.82** | **0.98** | **0.47** | **0.63** | 759.5 |

### Sorgu senaryosuna göre recall@10

| Arka uç | TTA | kopya | **kırpılmış** | orijinal | **farklı renk** | döndürülmüş |
|---|---|---|---|---|---|---|
| hash | kapalı | 0.38 | 0.00 | 0.48 | 0.46 | 0.06 |
| hash | açık | 0.52 | 0.04 | 0.71 | **0.65** | 0.71 |
| embedding | kapalı | 0.52 | 0.00 | 0.77 | 0.27 | 0.77 |
| embedding | açık | 0.50 | 0.00 | 0.75 | 0.25 | 0.75 |
| embedding + 3×3 karo | açık | 0.50 | **0.40** | 0.75 | 0.25 | 0.75 |
| **hibrit + 3×3 karo** | açık | 0.62 | **0.40** | 0.73 | **0.67** | 0.73 |

---

## Çıkarımlar

**1. Senaryo kırılımı şart.** Toplam recall@10'da hash+TTA (0.53) ile
embedding+karo (0.53) eşit görünüyor; kırılıma bakınca tamamen farklı işler
yaptıkları anlaşılıyor — biri colorway'de, diğeri kırpmada güçlü.

**2. Kırpılmış sorguları yalnızca karo indeksleme çözüyor.** Hash 0.00–0.04,
karosuz embedding 0.00, 3×3 karo ile 0.40. Karo indekslemenin varlık sebebi
budur; maliyeti (indeksleme ~10× yavaş) yalnızca bu senaryo için ödenir.

**3. Hash, colorway varyantlarında DINOv2'den iyi** (0.65 vs 0.25). Algısal
hash gri tonlama üzerinden çalıştığı için renk değişimine yapısal olarak
dayanıklıdır. Bu setin palet kurgusu farkı abartıyor olabilir, ama yönü
gerçektir ve hibrit tasarımını doğrudan belirledi.

**4. Hibrit, iki skorun maksimumunu almalı.** İlk uygulamada hibrit embedding
skorunu üste yazıyordu ve hash'in colorway üstünlüğünü kaybediyordu
(hit@10 0.85, recolor 0.25). Maksimum alınınca hem colorway hem kırpma
korunuyor: **hit@10 0.98, recolor 0.67** — sorgu süresi değişmeden.

**5. TTA hash'te büyük fark yaratıyor, embedding'de yaratmıyor.** Hash'te
recall@10 0.28 → 0.53 (döndürülmüş senaryosu 0.06 → 0.71). Embedding'de
0.47 → 0.45, yani ölçülebilir bir kazanç yok ama sorgu süresi 7× artıyor
(115 ms → 778 ms). DINOv2'nin döndürmeye zaten makul dayanıklı olması bunu
açıklıyor. Gerçek arşivde doğrulanırsa AI modunda TTA varsayılan olarak
kapatılabilir.

**6. Varsayılan arka uç hash olarak bırakıldı.** Hibrit en iyi sonucu veriyor
ama model indirmeyi (88 MB) ve karo indeksleme açıksa ~10× indeksleme süresini
gerektiriyor. İlk çalıştırmada bunu dayatmak yerine hash varsayılan kalıyor;
arayüz AI'yı öneriyor ve hibrit "önerilen" olarak işaretli.
