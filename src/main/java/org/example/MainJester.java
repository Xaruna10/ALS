package org.example;

import org.apache.commons.collections.CollectionUtils;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.*;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MainJester {
    public static Map<Integer, Set<Integer>> getWatchedMovieMap(List<Integer> userList, Dataset<Row> testData){
        Map<Integer, Set<Integer>> movieMap = new HashMap<>();
        for(int user: userList){
            List<Integer> watchedMovie = testData
                    .filter("userID = " + user)
                    .select("JokeID")
                    .as(Encoders.INT())
                    .collectAsList();
            movieMap.put(user, new HashSet<>(watchedMovie));
        }
        return movieMap;
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
        SparkSession spark = SparkSession.builder()
                .appName("ALS_test")
                .master("local[*]")
                .getOrCreate();
        String csvPath = "./src/main/java/org/example/JesterJokeDataset.csv";
        Dataset<Row> csvData = spark.read()
                .option("header","true")
                .option("inferSchema","true")
                .csv(csvPath);

        csvData.head();

        Dataset<Row> userCounts = csvData.groupBy("userID")
                .agg(functions.count("Rating").as("rating_count"));

        userCounts.show();

        Dataset<Row> activeUsers = userCounts.filter(userCounts.col("rating_count").gt(100));


        //Dataset<Row> activeUsers = userCounts.filter(userCounts.col("rating_count").gt(20));
        System.out.println("Active user count: "+ activeUsers.collectAsList().size());
        Dataset<Row> activeUserData = csvData.join(activeUsers, "userID");
        Dataset<Row> test = activeUserData.groupBy("userID").agg(functions.expr("percentile_approx(Rating, 0.2)").as("split_threshold"));

        test = activeUserData.join(test, "userID")
                .filter(activeUserData.col("Rating").leq(test.col("split_threshold")));

        test = test.drop("split_threshold");
        test = test.drop("rating_count");

        Dataset<Row> training = csvData.except(test);
////        Dataset<Row>[] splits = csvData.randomSplit(new double[]{0.8, 0.2});
////        Dataset<Row> training = splits[0];
////        Dataset<Row> test = splits[1];
        System.out.println("Testing count: "+ test.collectAsList().size());
        System.out.println("Training count: "+ training.collectAsList().size());

        int rank = 10; // Number of latent factors
        int numIterations = 10; // Number of iterations
        double lambda = 0.01;

        ALS als = new ALS()
                .setRegParam(lambda)
                .setMaxIter(numIterations)
                .setRank(rank)
                .setUserCol("userID")
                .setItemCol("JokeID")
                .setRatingCol("Rating")
                .setColdStartStrategy("drop")
                .setNonnegative(true);

        ALSModel alsModel = als.fit(training);
        System.out.println("Model training fitted");
        Dataset<Row> users = test.select(als.getUserCol()).distinct();
        List<Integer> userIds = users.select("userID").as(Encoders.INT()).distinct().collectAsList();

        Dataset<Row> movies = test.select(als.getItemCol()).distinct();
        List<Integer> movi = movies.select("JokeID").as(Encoders.INT()).distinct().collectAsList();

          Map<Integer, Map<Integer, Float>> userRecommendationsMap = new HashMap<>();
        long start = System.currentTimeMillis();
        System.out.println("Calulating user reccomendation map");
        for(int user: userIds){
            Dataset<Row> reccomend = alsModel.recommendForUserSubset(users.filter(users.col("userID").equalTo(user)), movi.size());
            //reccomend.show(false);

            reccomend.collectAsList().forEach(row -> {
                Integer userId = row.getAs("userID");
                List<Row> recommendations = row.getList(row.fieldIndex("recommendations"));
                Map<Integer, Float> temp = new HashMap<>();
                recommendations.forEach(rec -> {
                    Integer movieId = rec.getAs("JokeID");
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
//            FileOutputStream fileOut = new FileOutputStream("C:\\Users\\Checkout\\IdeaProjects\\ALS_Spark\\src\\main\\java\\org\\example\\JesterMap.ser");
//            ObjectOutputStream out = new ObjectOutputStream(fileOut);
//            out.writeObject(userRecommendationsMap);
//            out.close();
//            fileOut.close();
//            System.out.println("Serialized data is saved in userRecommendationsMap.ser");
//        } catch (IOException i) {
//            i.printStackTrace();
//        }
//        try {
//            FileInputStream fileIn = new FileInputStream("C:\\Users\\Checkout\\IdeaProjects\\ALS_Spark\\src\\main\\java\\org\\example\\JesterMap.ser");
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
//        }

        Map<Integer, Map<Integer, Float>> normalized = new HashMap<>(userRecommendationsMap);

        for (Map.Entry<Integer, Map<Integer, Float>> entry : normalized.entrySet()) {
            Integer outerKey = entry.getKey();
            Map<Integer, Float> innerMap = entry.getValue();
            for (Map.Entry<Integer, Float> innerEntry : innerMap.entrySet()) {
                Integer innerKey = innerEntry.getKey();
                Float value1 = innerEntry.getValue();
                if(value1<3.8){
                    normalized.get(outerKey).put(innerKey,0F);
                }
            }
        }
        System.out.println("3");
        Map<Integer, Set<Integer>> testDataMap = getWatchedMovieMap(userIds, test);
        System.out.println("4");
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
        System.out.println("4");
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
