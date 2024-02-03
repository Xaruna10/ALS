package org.example;

import org.apache.commons.collections.CollectionUtils;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.*;


import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {

    public static Map<Integer, List<Map<Integer, Float>>> movieRating = new HashMap<>();

    public static int count=0;
    public static float rootError = 0;

//    public static Map<Integer, Map<Integer, Float>> getWatchedMovieMap(List<Integer> userList, Dataset<Row> testData){
//        Map<Integer, Map<Integer, Float>> movieMap = new HashMap<>();
//        for(int user: userList){
//            Map<Integer, Float> temp = new HashMap<>();
//            testData.foreach(rec ->{
//                Integer userId = rec.getAs("userid");
//                Integer movieId = rec.getAs("movieid");
//                Integer rating = rec.getAs("movieRating");
//
//            });
//        }
//        return movieMap;
//    }
    public static Map<Integer, Set<Integer>> getWatchedMovieMap(List<Integer> userList, Dataset<Row> testData){
        Map<Integer, Set<Integer>> movieMap = new HashMap<>();
        for(int user: userList){
            List<Integer> watchedMovie = testData
                    .filter("userid = " + user)
                    .select("movieid")
                    .as(Encoders.INT())
                    .collectAsList();
            movieMap.put(user, new HashSet<>(watchedMovie));
        }
        return movieMap;
    }
    public static void addToMap(int userId, int movieId, float rating){
        Map<Integer, Float> temp =  new HashMap<>();
        if(rating>5){
            count++;
        }
        temp.put(movieId, rating);
        if(movieRating.containsKey(userId)){
            movieRating.get(userId).add(temp);
        }
        else{
            List<Map<Integer, Float>> tempList = new ArrayList<>();
            tempList.add(temp);
            movieRating.put(userId, tempList);
        }
    }
    public static double rmse(Map<Integer, Map<Integer, Float>> userRecommendationsMap, Dataset<Row> testingData, float maxRating, float minRating){
        float newMin = 1;
        float newMax = 5;
        testingData.foreach(rec -> {
                    Integer userId = rec.getAs("userid");
                    Integer movieId = rec.getAs("movieid");
                    Integer rating = rec.getAs("movieRating");
                    if(userId!=915 && userId != 516) {

                        System.out.println("Calculating ratings for user: " + userId + " and movie: " + movieId);
                        float predictedRating;
                        if (userRecommendationsMap.get(userId).keySet().contains(movieId)) {
                            predictedRating = userRecommendationsMap.get(userId).get(movieId);
                        } else {
                            predictedRating = 0F;
                        }

                        float normalizedRating = newMin + ((rating - minRating) * (newMax - newMin) / (maxRating - minRating));
                        userRecommendationsMap.get(userId).put(movieId,predictedRating);
                        rootError = rootError + (normalizedRating - predictedRating) * (normalizedRating - predictedRating);
                        count++;
                    }
                });
        return Math.sqrt(rootError/count);
    }

    public static List<Double> precison(Map<Integer, Map<Integer, Float>> userRecommendationsMap,Map<Integer, Set<Integer>> testDataMap, int n){
        List<Double> precisionList = new ArrayList<>();
        Map<Integer, List<Integer>> topMoviesForUsers = new HashMap<>();
        userRecommendationsMap.forEach((userId, moviesRatings) -> {
            // Create a stream from the movie ratings map entries
            Stream<Map.Entry<Integer, Float>> sortedMovies = moviesRatings.entrySet().stream()
                    //.filter(entry -> entry.getValue() > 0)
                    .sorted(Map.Entry.<Integer, Float>comparingByValue().reversed()) // Sort by rating in descending order
                    .limit(n); // Limit to top 5

            // Collect the top 5 movie IDs
            List<Integer> top5Movies = sortedMovies
                    .map(Map.Entry::getKey) // Get the movie ID from each entry
                    .collect(Collectors.toList());

            topMoviesForUsers.put(userId, top5Movies);
        });
        for(int user: topMoviesForUsers.keySet()){
            //System.out.println("calculating for user: "+ user+" where n="+ n);
            double commonCount;
            List<Integer> list1 = null;
            List<Integer> list2 = null;
            if(topMoviesForUsers.keySet().contains(user)) {
                if(topMoviesForUsers.get(user).size()>0) {
                    list1 = topMoviesForUsers.get(user);
                }
                if(testDataMap.get(user)!=null && testDataMap.get(user).size()>0) {
                    list2 = testDataMap.get(user).stream().toList();
                }
                if(list1==null || list2==null || list2.size()==0 || list1.size()==0){
                    commonCount=0;
                }
                else {
                    commonCount = CollectionUtils.intersection(list1, list2).size();
                }
            }
            else {
                commonCount=0;
            }
            //double commonCount = CollectionUtils.intersection(topMoviesForUsers.get(user),testDataMap.get(user).stream().toList()).size();
            //precisionMap.put(user, commonCount/n);
            precisionList.add(commonCount/n);
        }
        return precisionList;
    }
    public static List<Double> recall(Map<Integer, Map<Integer, Float>> userRecommendationsMap,Map<Integer, Set<Integer>> testDataMap, int n){
        List<Double> recallList = new ArrayList<>();
        Map<Integer, List<Integer>> topMoviesForUsers = new HashMap<>();
        userRecommendationsMap.forEach((userId, moviesRatings) -> {
            // Create a stream from the movie ratings map entries
            Stream<Map.Entry<Integer, Float>> sortedMovies = moviesRatings.entrySet().stream()
                    .filter(entry -> entry.getValue() > 0)
                    .sorted(Map.Entry.<Integer, Float>comparingByValue().reversed()) // Sort by rating in descending order
                    .limit(n); // Limit to top 5

            // Collect the top 5 movie IDs
            List<Integer> top5Movies = sortedMovies
                    .map(Map.Entry::getKey) // Get the movie ID from each entry
                    .collect(Collectors.toList());

            topMoviesForUsers.put(userId, top5Movies);
        });
        for(int user: topMoviesForUsers.keySet()){
            //System.out.println("calculating for user: "+ user+" where n="+ n);
            double commonCount;
            List<Integer> list1 = null;
            List<Integer> list2 = null;
            if(topMoviesForUsers.keySet().contains(user) && topMoviesForUsers.get(user)!=null) {
                //commonCount = CollectionUtils.intersection(topMoviesForUsers.get(user), testDataMap.get(user).stream().toList()).size();
                if(topMoviesForUsers.get(user).size()>0) {
                    list1 = topMoviesForUsers.get(user);
                }
                if(testDataMap.get(user)!=null && testDataMap.get(user).size()>0) {
                    list2 = testDataMap.get(user).stream().toList();
                }
                if(list1==null || list2==null || list2.size()==0 || list1.size()==0){
                    commonCount=0;
                }
                else {
                    commonCount = CollectionUtils.intersection(list1, list2).size();
                }
            }
            else {
                commonCount=0;
            }
            //precisionMap.put(user, commonCount/n);
            if (commonCount==0){
                recallList.add(0D);
            }
            else {
                recallList.add(commonCount / testDataMap.get(user).size());
            }
        }
        return recallList;
    }
    public static List<Double> F1score(List<Double> precision, List<Double> recall){
        List<Double> f1List =new ArrayList<>();
        double f1=0;
        for(int i=0;i<Math.max(precision.size(), recall.size());i++){
            f1 = (precision.get(i) + recall.get(i) > 0) ? 2 * precision.get(i) * recall.get(i) / (precision.get(i) + recall.get(i)) : 0;
            f1List.add(f1);
        }
        return f1List;
    }
    public static double calculateAverage(List<Double> values) {
        double sum = 0;
        if(!values.isEmpty()) {
            for (Double value : values) {
                sum += value;
            }
            return sum / values.size();
        }
        return sum;
    }

    public static void main(String[] args) {
        // create a spark session
        //Spark will run in local mode with one worker thread, which effectively restricts the parallelism of Spark operations to a single batch.
        SparkSession spark = SparkSession.builder()
                .appName("ALS_test")
                .master("local[*]")
                .getOrCreate();

        String csvPath = "./src/main/java/org/example/1m.csv";
        Dataset<Row> csvData = spark.read()
                .option("header","true")
                .option("inferSchema","true")
                .csv(csvPath);

        //Dataset<Row> distinctMovieIDs = csvData.select("movieid").distinct();
        //long uniqueMovieId = distinctMovieIDs.count();
        // Test Train Split
//        Dataset<Row>[] splits = csvData.randomSplit(new double[]{0.8, 0.2});
//        Dataset<Row> trainingData = splits[0];
//        Dataset<Row> testingData = splits[1];

        //Dataset<Row> userCounts = csvData.groupBy("userId").count().filter("count > 1");

        // Collect user IDs with multiple entries
        // Collect user IDs with multiple entries

        Dataset<Row> userCounts = csvData.groupBy("userid")
                .agg(functions.count("movieRating").as("rating_count"));

        Dataset<Row> activeUsers = userCounts.filter(userCounts.col("rating_count").gt(100));

        Dataset<Row> activeUserData = csvData.join(activeUsers, "userid");
        Dataset<Row> test = activeUserData.groupBy("userid").agg(functions.expr("percentile_approx(movieRating, 0.2)").as("split_threshold"));

        test = activeUserData.join(test, "userid")
                .filter(activeUserData.col("movieRating").leq(test.col("split_threshold")));

        // Now 'test' contains 20% of the rows of each active user.

        test = test.drop("split_threshold");
        test = test.drop("rating_count");
//        System.out.println(test.schema().toString());
//        test.head(3);
        Dataset<Row> training = csvData.except(test);

        System.out.println("Testing count: "+ test.collectAsList().size());
        System.out.println("Training count: "+ training.collectAsList().size());
//        System.out.println("Traning count: "+ trainingData.collectAsList().size());
//        System.out.println("Testing size: "+ testingData.collectAsList().size());



        // Find users with multiple entries
//        Dataset<Row> userCounts = csvData.groupBy("userId").count().filter("count > 1");
//
//        // Collect user IDs with multiple entries
//        List<Integer> usersWithMultipleEntries = userCounts.select("userId")
//                .as(Encoders.INT())
//                .collectAsList()
//                .stream()
//                .collect(Collectors.toList());

//        System.out.println("Data split is done for test and train");

        // Build the recommendation model using ALS on the training data
        int rank = 10; // Number of latent factors
        int numIterations = 10; // Number of iterations
        double lambda = 0.01;

//        System.out.println("Model tarinng Started");
        ALS als = new ALS()
                .setRegParam(lambda)
                .setMaxIter(numIterations)
                .setRank(rank)
                .setUserCol("userid")
                .setItemCol("movieid")
                .setRatingCol("movieRating")
                .setColdStartStrategy("drop")
                .setNonnegative(true);

//        long start = System.currentTimeMillis();

        ALSModel alsModel = als.fit(training);
        // attempt 1
//        Dataset<Row> userMoviePairs = csvData.select("userid").distinct()
//                .crossJoin(csvData.select("movieid").distinct());
//
//        Dataset<Row> predictions = alsModel.transform(userMoviePairs);
//        predictions.show();
//
//
//        predictions.foreach(row ->{
//            int userId = row.getAs("userid");
//            int movieId = row.getAs("movieid");
//            float rating = row.getAs("prediction");
//            System.out.println("The user "+ userId+" has rated movie "+movieId+" as "+rating);
//            addToMap(userId, movieId, rating);
//        });

        //attemp 2
        Dataset<Row> users = test.select(als.getUserCol()).distinct();
        List<Integer> userIds = users.select("userId").as(Encoders.INT()).distinct().collectAsList();

        Dataset<Row> movies = test.select(als.getItemCol()).distinct();
        List<Integer> movi = movies.select("movieId").as(Encoders.INT()).distinct().collectAsList();

        Map<Integer, Map<Integer, Float>> userRecommendationsMap = new HashMap<>();
        long start = System.currentTimeMillis();
        for(int user: userIds){
            Dataset<Row> reccomend = alsModel.recommendForUserSubset(users.filter(users.col("userId").equalTo(user)), 1682);
            //reccomend.show(false);

            reccomend.collectAsList().forEach(row -> {
                Integer userId = row.getAs("userid");
                List<Row> recommendations = row.getList(row.fieldIndex("recommendations"));
                Map<Integer, Float> temp = new HashMap<>();
                recommendations.forEach(rec -> {
                    Integer movieId = rec.getAs("movieid");
                    Float rating = rec.getAs("rating");
                    float clippedRating = Math.max(1, Math.min(rating, 5));
                    temp.put(movieId, clippedRating);
                    //userRecommendationsMap.putAll(temp);
                });
                //System.out.println("For user: "+ user+" getting total movies as: "+temp.size());
                for(int mov: movi){
                    if(!temp.keySet().contains(mov)){
                        System.out.println("User "+user+" has no movie rating for "+ mov);
                        temp.put(mov,0F);
                    }
                }
                userRecommendationsMap.put(userId, temp);
            });
        }

        long end = System.currentTimeMillis();
        System.out.println("Time elapsed: "+ (end-start));
//        try {
//            FileOutputStream fileOut = new FileOutputStream("C:\\Users\\Checkout\\IdeaProjects\\ALS_Spark\\src\\main\\java\\org\\example\\userRecommendationsMap2.ser");
//            ObjectOutputStream out = new ObjectOutputStream(fileOut);
//            out.writeObject(userRecommendationsMap);
//            out.close();
//            fileOut.close();
//            System.out.println("Serialized data is saved in userRecommendationsMap.ser");
//        } catch (IOException i) {
//            i.printStackTrace();
//        }


        //Map<Integer, Map<Integer, Float>> userRecommendationsMap = null;

//        try {
//            FileInputStream fileIn = new FileInputStream("C:\\Users\\Checkout\\IdeaProjects\\ALS_Spark\\src\\main\\java\\org\\example\\userRecommendationsMap.ser");
//            ObjectInputStream in = new ObjectInputStream(fileIn);
//            userRecommendationsMap = (Map<Integer, Map<Integer, Float>>) in.readObject();
//            in.close();
//            fileIn.close();
//        } catch (IOException i) {
//            i.printStackTrace();
//            return;
//        } catch (ClassNotFoundException c) {
//            System.out.println("Map class not found");
//            c.printStackTrace();
//
//        }
//        for (int user: userIds){
//            if(!userRecommendationsMap.keySet().contains(user)){
//                System.out.println("user: "+ user +" not present");
//            }
//        }
        float maxRating = Float.MIN_VALUE;  // Initialize to the smallest possible float value
        float minRating = Float.MAX_VALUE;  // Initialize to the largest possible float value

        for (Map<Integer, Float> userRatings : userRecommendationsMap.values()) {
            for (Float rating : userRatings.values()) {
                if (rating > maxRating) {
                    maxRating = rating;
                }
                if (rating < minRating) {
                    minRating = rating;
                }
            }
        }
//        for(int user: userRecommendationsMap.keySet()){
//            System.out.println("for user: "+user+ " the movie count is: "+userRecommendationsMap.get(user).size());
//        }

        //System.out.println("RMSE value is : "+ rmse(userRecommendationsMap, testingData, maxRating, 0));

//        String csvFile = "C:\\Users\\Checkout\\IdeaProjects\\ALS_Spark\\src\\main\\java\\org\\example\\data1.csv";
//        try (FileWriter writer = new FileWriter(csvFile)) {
//            for (Map.Entry<Integer, Map<Integer, Float>> entry : userRecommendationsMap.entrySet()) {
//                Integer outerKey = entry.getKey();
//                Map<Integer, Float> innerMap = entry.getValue();
//                for (Map.Entry<Integer, Float> innerEntry : innerMap.entrySet()) {
//                    Integer innerKey = innerEntry.getKey();
//                    Float value = innerEntry.getValue();
//                    if(value>5){
//                        System.out.println("for user: "+ outerKey+ " and movie: "+innerKey);
//                    }
//                    String line = outerKey + "," + innerKey + "," + value + "\n";
//                    writer.write(line);
//                }
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        Map<Integer, Map<Integer, Float>> normalized = new HashMap<>(userRecommendationsMap);

        for (Map.Entry<Integer, Map<Integer, Float>> entry : normalized.entrySet()) {
            Integer outerKey = entry.getKey();
            Map<Integer, Float> innerMap = entry.getValue();
            for (Map.Entry<Integer, Float> innerEntry : innerMap.entrySet()) {
                Integer innerKey = innerEntry.getKey();
                Float value = innerEntry.getValue();
                if(value<3.8){
                    normalized.get(outerKey).put(innerKey,0F);
                }
            }
        }

        Map<Integer, Set<Integer>> testDataMap = getWatchedMovieMap(userIds, test);

        List<Double> precisionList = precison(normalized, testDataMap, 5);
        System.out.println("For n=5 Precison is: "+ calculateAverage(precisionList));
        List<Double> precisionList1 = precison(normalized, testDataMap, 10);
        System.out.println("For n=10 Precison is: "+ calculateAverage(precisionList1));
        List<Double> precisionList2 = precison(normalized, testDataMap, 25);
        System.out.println("For n=25 Precison is: "+ calculateAverage(precisionList2));
        List<Double> precisionList3 = precison(normalized, testDataMap, 50);
        System.out.println("For n=50 Precison is: "+ calculateAverage(precisionList3));
        List<Double> precisionList4 = precison(normalized, testDataMap, 100);
        System.out.println("For n=100 Precison is: "+ calculateAverage(precisionList4));

        //recall
        List<Double> recallList = recall(normalized, testDataMap, 5);
        System.out.println("For n=5 Recall is: "+ calculateAverage(recallList));
        List<Double> recallList1 = precison(normalized, testDataMap, 10);
        System.out.println("For n=10 Recall is: "+ calculateAverage(recallList1));
        List<Double> recallList2 = precison(normalized, testDataMap, 25);
        System.out.println("For n=25 Recall is: "+ calculateAverage(recallList2));
        List<Double> recallList3 = precison(normalized, testDataMap, 50);
        System.out.println("For n=50 Recall is: "+ calculateAverage(recallList3));
        List<Double> recallList4 = precison(normalized, testDataMap, 100);
        System.out.println("For n=100 Recall is: "+ calculateAverage(recallList4));

        //F1 score
        List<Double> F1List = F1score(precisionList, recallList);
        System.out.println("For n=5 F1 score is: "+calculateAverage(F1List));
        List<Double> F1List1 = F1score(precisionList1, recallList1);
        System.out.println("For n=10 F1 score is: "+calculateAverage(F1List1));
        List<Double> F1List2 = F1score(precisionList2, recallList2);
        System.out.println("For n=25 F1 score is: "+calculateAverage(F1List2));
        List<Double> F1List3 = F1score(precisionList3, recallList3);
        System.out.println("For n=50 F1 score is: "+calculateAverage(F1List3));
        List<Double> F1List4 = F1score(precisionList4, recallList4);
        System.out.println("For n=5 F1 score is: "+calculateAverage(F1List4));
        spark.stop();
    }
}


// show user that have no data predicted
// show movies for user that have no d\ata predicted
