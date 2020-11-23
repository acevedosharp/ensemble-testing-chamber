# testing-chamber
Run the following commands and that's it! :)

1. sudo apt update && sudo apt upgrade
2. sudo apt install curl
3. curl https://storage.googleapis.com/testing-chamber-bundle/bundle.zip -o bundle.zip
4. sudo apt-get install zip gzip tar
5. unzip bundle.zip
6. cd bundle
7. sudo apt install default-jre

// For HASCO runner

8.1. java -Xmx15G -jar testing-chamber-1.0.jar 60 5 4 true false datasets/dexter.arff datasets/madelon.arff datasets/dorothea.arff datasets/amazon-commerce-reviews.arff datasets/convex.arff

// For MLPlan runner

8.2. java -Xmx15G -jar testing-chamber-1.0.jar 60 5 4 false true datasets/dexter.arff datasets/madelon.arff datasets/dorothea.arff datasets/amazon-commerce-reviews.arff datasets/convex.arff
