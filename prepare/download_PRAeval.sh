
# PRA classifier code and weights

FILE_ID="1z4uwyX6Iq-AmSsUhDKm6EKdJCGFDhOCd"
DEST_DIR="../dependency"

ZIP_FILE="./PRA_classifier.zip"

echo "Downloading PRA_classifier.zip from Google Drive"
gdown --id $FILE_ID -O $ZIP_FILE

echo "Unzipping..."
unzip -q $ZIP_FILE -d $DEST_DIR
rm -f $ZIP_FILE
