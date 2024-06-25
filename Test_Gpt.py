import Gpt_summarise

mail = """
株式会社●●●●
●●●●様

お世話になっております。
一般社団法人日本ビジネスメール協会の山田太郎です。

●月●日（●）●時のお打ち合わせ場所について、
ご連絡いたします。

--------------------------------
■場所
一般社団法人日本ビジネスメール協会

■住所
東京都千代田区神田小川町2-1 KIMURA BUILDING 5階

■交通アクセス
都営新宿線「小川町」駅 徒歩1分
東京メトロ丸の内線「淡路町」駅 徒歩2分
東京メトロ千代田線「新御茶ノ水」駅 徒歩2分
JR「御茶ノ水」駅 徒歩10分
JR「神田」駅 徒歩11分

■地図
https://businessmail.or.jp/about/access
--------------------------------

近隣で迷う方が多いため、何かございましたらお電話ください。
（電話：03-5577-3210）

それでは当日、●●様のお越しをお待ちしております。

よろしくお願いいたします。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一般社団法人日本ビジネスメール協会　山田 太郎（YAMADA Taro）
〒101-0052 東京都千代田区神田小川町2-1 KIMURA BUILDING 5階
電話 03-5577-3210 / FAX 03-5577-3238 / メール info@businessmail.or.jp
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一般社団法人日本ビジネスメール協会　https://businessmail.or.jp/
アイ・コミュニケーション公式サイト　http://www.sc-p.jp/
ビジネスメールの教科書　https://business-mail.jp/
"""

api_key = "sk-proj-eBwCmFJCxyTto80BZgfiT3BlbkFJsfe1ODtGFKgMkRVTh9d8"
print(Gpt_summarise.summarise_email(email_content=mail,api_key=api_key))