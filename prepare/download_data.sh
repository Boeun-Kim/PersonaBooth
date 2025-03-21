

# PerMo SMPL
FILE_ID="1ZCiOVVQAdHAOeOJttlWFw1-YrNoSesga"
DEST_DIR="../data/dataset/smpl"

ZIP_FILE="./PerMo_smpl.zip"

echo "Downloading PerMo_smpl.zip from Google Drive"
gdown --id $FILE_ID -O $ZIP_FILE

echo "Unzipping..."
unzip -q $ZIP_FILE -d $DEST_DIR
rm -f $ZIP_FILE

# PerMo Description
FILE_ID="1wMpTVbi7bKqa74SRYfMUyUNvasvHRNuU"
DEST_DIR="../data/dataset/description"

ZIP_FILE="./PerMo_description.zip"

echo "Downloading PerMo_description.zip from Google Drive"
gdown --id $FILE_ID -O $ZIP_FILE

echo "Unzipping..."
unzip -q $ZIP_FILE -d $DEST_DIR
rm -f $ZIP_FILE