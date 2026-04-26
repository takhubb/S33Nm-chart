# J-Quants API V2 charts

J-Quants API V2 から過去 5 年分の週次データを取得し、次のチャートを作成します。

- 業種別の相対株価チャート。全業種を 1 枚にまとめ、TOPIX を重ねます。
- 市場別の相対株価チャート。プライム、スタンダード、グロースを 1 枚にまとめ、TOPIX を重ねます。
- 業種ごとの PER / PBR 推移チャート。

## Setup

```bash
pyenv local 3.13.13
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env` に J-Quants API V2 の API キーを入れてください。

```bash
JQUANTS_API_KEY=your_api_key
```

## Run

```bash
source .venv/bin/activate
python scripts/generate_charts.py
```

API のレート制限が出る場合は、デフォルトのまま再実行してください。取得済みの日付は `data/cache/` から再利用されます。

出力先:

- `output/charts/industry_relative_weekly.png`
- `output/charts/market_relative_weekly.png`
- `output/charts/sector_per_pbr/*.png`
- `output/data/*.csv`

## Method

業種別・市場別のチャートは、個別銘柄の週末営業日の調整終値を使い、銘柄ごとに最初の価格を 100 としたうえで分類ごとの中央値を取っています。さらにグラフ上の各系列も最初の値を 100 にそろえています。平均で集計したい場合は `--group-agg mean` を指定できます。

市場区分と業種区分は、取得期間末時点の上場銘柄マスタで分類しています。TOPIX は J-Quants API V2 の TOPIX 日足を週次化して重ねています。

PER は直近開示済みの予想 EPS を優先し、なければ次期予想 EPS、実績 EPS の順で使います。PBR は直近開示済みの BPS を使います。業種別 PER / PBR は個別銘柄の倍率の中央値です。

PER / PBR はデフォルトで 3 社以上のデータがある週だけ描画します。変更する場合は `--min-valuation-count` を指定してください。
