import mail_classification 

# 新しいメッセージ
new_message = """
留学生チューター担当者 各位

情報学研究科教務掛です。

いつも本研究科留学生へのチューター業務にご尽力いただき、
ありがとうございます。

標記の件につきまして。国際教育交流課より通知がありました。
添付の資料を参照のうえ、ご利用ください。
留学生へはKULASISにて周知しています。

概要：
例年（公財）京都市国際交流協会より発行されている「留学生おこしやすPASS」が 
今年度もスマホ等の画面にて表示できる画像として配布されることになりました。
この事業は、本学の留学生と引率の日本人京大生（※同数以上の留学生同伴の場合に限る）
に対する優待プログラムです。（※利用時、学生証の提示要。）
"""
model,scaler = mail_classification.load_model("model.pickle","scaler.pickle")

#カテゴリごとの確率分布を取得
predicted_proba = mail_classification.classify_mail(new_message,model,scaler) 

print('Predicted Probabilities:', predicted_proba)