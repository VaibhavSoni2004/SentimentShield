-- SELECT DISTINCT sentiment, COUNT(sentiment) as sentiment_count
-- FROM feedback_logs
-- GROUP BY sentiment
-- ORDER BY sentiment_count DESC;

-- select * from feedback_logs
-- where sentiment = 'Positive';

select status from feedback_logs;
