
# AMASS body model
# These files are originally from https://github.com/nghorbani/amass?tab=readme-ov-file#body-models

FILE_ID="15djesGGB58AE-78HMe6PW-0QX7A3iDq9"
DEST_DIR="../data"

ZIP_FILE="./body_models.zip"

echo "Downloading body_models.zip from Google Drive"
gdown --id $FILE_ID -O $ZIP_FILE

echo "Unzipping..."
unzip -q $ZIP_FILE -d $DEST_DIR
rm -f $ZIP_FILE