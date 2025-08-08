# AIOCR を Cloud Run にデプロイする手順（GCP初心者向け）

本手順では、コンテナ化済みのAIOCRアプリを Cloud Run にデプロイします。OpenAI APIキーは Secret Manager を利用し、アイデンティティ連携（IAPなしの簡易公開 or Googleアカウント限定）までカバーします。

## 前提
- GCP プロジェクトを作成済み（以降 `PROJECT_ID`）
- 請求先（Billing）有効化済み
- Cloud Run / Artifact Registry / Secret Manager API 有効化済み
- gcloud CLI セットアップ済み（`gcloud init`）

## 1. ソース取得とビルド
```bash
# プロジェクト設定
PROJECT_ID=your-project-id
REGION=asia-northeast1  # 東京など任意
SERVICE=aiocr-app

# 認証
gcloud auth login
gcloud config set project $PROJECT_ID

# Artifact Registry リポジトリ作成（初回のみ）
gcloud artifacts repositories create containers \
  --repository-format=docker \
  --location=$REGION \
  --description="AIOCR containers"

# Cloud Build でコンテナビルド＆登録
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/containers/$SERVICE:latest
```

## 2. Secret Manager（OpenAI APIキー）
```bash
# シークレットを作成（すでにある場合はスキップ）
printf "%s" "sk-xxxxx" | gcloud secrets create OPENAI_API_KEY --data-file=-

# ランタイムSAに参照権限を付与（後述のSA名に合わせる）
SA=cloud-run-aiocr@$PROJECT_ID.iam.gserviceaccount.com
gcloud secrets add-iam-policy-binding OPENAI_API_KEY \
  --member=serviceAccount:$SA \
  --role=roles/secretmanager.secretAccessor
```

## 3. サービスアカウント（実行用）
```bash
SA=cloud-run-aiocr@$PROJECT_ID.iam.gserviceaccount.com

gcloud iam service-accounts create cloud-run-aiocr \
  --display-name="AIOCR Cloud Run SA"

# 必要権限（最小）
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:$SA \
  --role=roles/run.invoker

# Secret Manager 参照権限（上で付与済み）
# Cloud Storage/Cloud SQLを使う場合は各サービスのロールも付与
```

## 4. デプロイ
```bash
REGION=asia-northeast1
IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/containers/$SERVICE:latest
SA=cloud-run-aiocr@$PROJECT_ID.iam.gserviceaccount.com

# Secret 環境変数として注入（Cloud Runの新UIなら管理画面から設定でもOK）
gcloud run deploy $SERVICE \
  --image $IMAGE \
  --region $REGION \
  --service-account $SA \
  --allow-unauthenticated \
  --update-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest \
  --port 8080

# 公開URL取得
gcloud run services describe $SERVICE --region $REGION --format='value(status.url)'
```

- 既定で誰でもアクセス可能（`--allow-unauthenticated`）。
- Googleアカウント限定公開にする場合は 5 章へ。

## 5. アクセス制御（認証が必要な場合）
- 「Googleアカウント（社内/指定アカウント）のみ許可」
  1) サービスを「認証が必要」に変更
  ```bash
  gcloud run services update $SERVICE --region $REGION --no-allow-unauthenticated
  ```
  2) 招待するユーザー/グループに `Cloud Run Invoker` を付与
  ```bash
  gcloud run services add-iam-policy-binding $SERVICE \
    --region $REGION \
    --member=user:someone@example.com \
    --role=roles/run.invoker
  ```
  3) アクセス時、Googleアカウントでのログインを求められます

- IAP（Identity-Aware Proxy）で厳格な制御をしたい場合は、外部HTTP(S)ロードバランサ＋IAPを構成します（本ガイド外）。

## 6. よくあるエラーと対処
- 403: Permission denied on secret
  - サービスアカウントに `roles/secretmanager.secretAccessor` が付与されていない
- 403: The user does not have permission to access the service
  - `roles/run.invoker` の付与漏れ（ユーザーまたはグループ）
- 起動しない: `PORT` 不一致
  - 本アプリは `$PORT` を使用。`Dockerfile`は `EXPOSE 8080` かつ `--server.port=${PORT}` で起動済み
- 画像処理エラー（libGL）
  - Cloud Runのコンテナ内は`libgl1`等が必要。`Dockerfile` で `libgl1` をインストール済み

## 7. 更新/再デプロイ
```bash
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/containers/$SERVICE:latest
gcloud run deploy $SERVICE --image $REGION-docker.pkg.dev/$PROJECT_ID/containers/$SERVICE:latest --region $REGION
```

## 8. オプション（ストレージやDBのマネージド化）
- workspace/テンポラリを Cloud Storage バケットへ
- SQLite を Cloud SQL(PostgreSQL/MySQL) へ
- これらはアプリ側の実装追加が必要です（ご要望があれば対応します）
