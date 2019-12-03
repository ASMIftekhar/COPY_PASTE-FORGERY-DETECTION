wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WzTNodYUgi6LfV0czs3uDvb4Yr0K9dzQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WzTNodYUgi6LfV0czs3uDvb4Yr0K9dzQ" -O pre.zip && rm -rf /tmp/cookies.txt

virtualenv -p /usr/bin/python3.6 copy_move
source copy_move/bin/activate
pip3 -r install requirements.txt



