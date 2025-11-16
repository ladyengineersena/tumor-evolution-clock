# Ä°laÃ§ EtkileÅŸimi Tespiti (Drug Interaction Extraction)

Bu proje, doÄŸal dil iÅŸleme (NLP) teknikleri kullanarak metinlerden ilaÃ§ etkileÅŸimlerini tespit etmeyi ve Ã§Ä±karmayÄ± amaÃ§lamaktadÄ±r.

## Ã–zellikler

- TÃ¼rkÃ§e metin desteÄŸi
- UTF-8 karakter encoding desteÄŸi
- Named Entity Recognition (NER) ile ilaÃ§ isimlerini bulma
- Ä°liÅŸki Ã§Ä±karÄ±mÄ± (relationship extraction) ile ilaÃ§ etkileÅŸimlerini tespit etme

## Kurulum

pip install -r requirements.txt

## KullanÄ±m

from ilac_etkilesimi import IlacEtkilesimTespiti
tespit = IlacEtkilesimTespiti()
etkilesimler = tespit.analiz_et("Aspirin ve warfarin birlikte kullanÄ±ldÄ±ÄŸÄ±nda kanama riski artar.")
print(etkilesimler)
