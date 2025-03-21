

# Pertrained PersonaBooth on PerMo dataset
FILE_ID="1rLQ6o2ZtQE2yI_R1qP2sMHDPwtiqFMSt"
DEST_DIR="../pretrained/PerMo_checkpoint.pt"

echo "Downloading PerMo_checkpoint.pt from Google Drive"
gdown --id $FILE_ID -O $DEST_DIR


# Pretrained TMR
FILE_ID="1dV5cNt041APe2egjKjTfscRxZBW7CEq7"
DEST_DIR="../dependency/TMR/pretrained"

ZIP_FILE="./TMR_weights.zip"

echo "Downloading TMR_weights.zip from Google Drive"
gdown --id $FILE_ID -O $ZIP_FILE

echo "Unzipping..."
unzip -q $ZIP_FILE -d $DEST_DIR
rm -f $ZIP_FILE