# azure_final_test
# 測試環境設定
1. [建議] 先建立虛擬環境
2. 在虛擬環境中安裝需要的套件

   `pip install -r requirements.txt`

# 修改環境變數
將 Azure AI 翻譯工具、電腦視覺等對應的區域、金鑰、服務端點寫入環境變數檔案(`.env`)

# 測試網頁頁用程式(web.py)
1. 執行 web.py

   `python web.py`
2. 開啟瀏覽器並前往 http://127.0.0.1:8080

# 部署到雲端
1. 在Azure建立容器登錄
2. 登入容器
3. 在該容器建立映像檔並上傳
4. 在Azure建立容器個體來執行該映像檔
