# testing-chamber
Run the following commands and that's it! :)

sudo apt update && sudo apt upgrade
sudo apt install curl
curl https://storage.googleapis.com/testing-chamber-bundle/bundle.zip -o bundle.zip

sudo apt-get install zip gzip tar
unzip bundle.zip
cd bundle

sudo apt install default-jre

// For HASCO runner
java -Xmx15G -jar testing-chamber-1.0.jar 60 5 4 true false datasets/dexter.arff datasets/madelon.arff datasets/dorothea.arff datasets/amazon-commerce-reviews.arff datasets/convex.arff

// For MLPlan runner
java -Xmx15G -jar testing-chamber-1.0.jar 60 5 4 false true datasets/dexter.arff datasets/madelon.arff datasets/dorothea.arff datasets/amazon-commerce-reviews.arff datasets/convex.arff
