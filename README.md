# testing-chamber
## An example to run on a 4 cores and 16 gigs machine
Run the following commands and that's it! :)

1. sudo apt update && sudo apt upgrade
2. sudo apt install curl
3. curl https://storage.googleapis.com/testing-chamber-bundle/bundle.zip -o bundle.zip
4. sudo apt-get install zip gzip tar
5. unzip bundle.zip
6. cd bundle
7. sudo apt install default-jre
8. java -Xms14G -Xmx14G -jar testing-chamber-1.0.jar 60 5 4 true false datasets/dexter.arff datasets/madelon.arff datasets/dorothea.arff datasets/amazon-commerce-reviews.arff datasets/convex.arff

// Options for fat jar are: minutes, repetitions, cores, runHasco, runMLPlan, datasets...


## Create bundle
1. gradle clean
2. gradle jar
3. move jar to directory with search space and datasets
4. zip it
5. upload to a bucket