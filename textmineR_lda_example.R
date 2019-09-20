library(dplyr)
library(stringr)
library(textmineR)
library(tidytext)
library(ggplot2)
library(tidyr)
library(purrr)

data(movie_review, package = "text2vec")

#Some initial cleaning of the text
movie_review$cleaned_review <- str_replace_all(movie_review$review, "\\d", "") #remove numbers

movie_review$cleaned_review <- str_replace_all(movie_review$cleaned_review, "(\\<br\\s*\\/\\>)", "") #get rid of HTML line breaks

##Create a filtered document-term matrix
df <- movie_review %>% 
  select(id, cleaned_review) %>% 
  unnest_tokens(output = word,
                input = cleaned_review,
                stopwords = c(stopwords::stopwords("en"),
                              stopwords::stopwords(source = "smart")),
                token = "ngrams",
                n_min = 1, n = 2) %>% 
  mutate(word = SnowballC::wordStem(word)) %>%
  count(id, word, name = "doc_count") %>% #needed for document-term matrix
  ungroup()

word_freq_df <- df %>% 
  count(word, name = "corpus_count") %>% 
  filter(corpus_count > 50) %>% #Keeping the dtm manageable. Some judgment involved here
  arrange(desc(corpus_count))

df_filtered <- df %>% 
  inner_join(word_freq_df, by = "word") %>% 
  select(id, word, doc_count)

#Cast as document-term matrix
df_dtm <- df_filtered %>% 
  cast_sparse(id, word, doc_count)

##Run multiple topic models to see which K makes the most sense
k_list <- seq(5, 50, by = 5)

# Fit a bunch of LDA models; this takes a while to run

model_list <- TmParallelApply(X = k_list, FUN = function(k){
  
  m <- FitLdaModel(dtm = df_dtm,
                   k = k,
                   iterations = 200,
                   burnin = 180,
                   alpha = 0.1,
                   beta = colSums(df_dtm) / sum(df_dtm) * 100,
                   optimize_alpha = TRUE,
                   calc_likelihood = FALSE,
                   calc_coherence = TRUE,
                   calc_r2 = TRUE,
                   cpus = 1)
  
  m$k <- k
  
  m
}, #export = ls(),
cpus = 6)

## Evaluation metrics

model_compare <- map_dfr(model_list, magrittr::extract, c("k", "r2")) %>% 
  mutate(coherence = map_dbl(model_list, function(x) mean(x$coherence)),
         topic_model = pluck(model_list))

model_compare %>% 
  transmute(k, 
            `R Squared` = r2,
            `Probablistic Coherence` = coherence) %>% 
  gather(Metric, Value, -k) %>% 
  ggplot(aes(k, Value, color = Metric)) +
  geom_line(size = 1.5, alpha = 0.7, show.legend = FALSE) +
  facet_wrap(~Metric, scales = "free_y") +
  labs(x = "K (number of topics)",
       y = NULL,
       title = "Model diagnostics by number of topics",
       subtitle = "45 topics seems optimal")

# model_list <- saveRDS('yadada') #Save entire set of models in case I change my mind later

#Choose the model with the optimal K to complete the analysis (save this object for later as well)
final_model <- model_compare %>% 
  filter(k == 45) %>% 
  pull(topic_model) %>% 
  .[[1]]

## Look more at this specific topic model, using some textmineR utilities

final_model$terms <- GetTopTerms(phi = final_model$phi, M = 5)

final_model$prevalence <- colSums(final_model$theta)/sum(final_model$theta) * 100

final_model$labels <- LabelTopics(assignments = final_model$theta > 0.05,
                                  dtm = df_dtm,
                                  M = 1)

eval_matrix <- data.frame(topic = rownames(final_model$phi),
                          label = final_model$label,
                          coherence = round(final_model$coherence, 3),
                          top_terms = apply(final_model$terms, 2, function(x){
                            paste(x, collapse = ", ")
                          }),
                          stringsAsFactors = F)

# Get a data frame with every topic probability (where gamma > 0.05) for every
# document, along with metadata

tidy_gamma <- data.frame(document = rownames(final_model$theta),
                         final_model$theta,
                         stringsAsFactors = F) %>% 
  gather(topic, gamma, -document) %>% 
  group_by(document) %>% 
  filter(gamma > 0.05) %>% 
  top_n(5, gamma) %>% 
  inner_join(eval_matrix, by = c("topic" = "topic")) %>% 
  select(document, topic, gamma, top_terms) %>% 
  arrange(desc(gamma)) %>% 
  mutate(g_row = row_number(), #I use g_row == 1 as filter in most analysis to find most probable topic per document
         topic = as.numeric(str_replace_all(topic, "t_", ""))) %>% 
  ungroup()

# join back to original data set

text_data_topics <- tidy_gamma %>% 
  left_join(movie_review, by = c("document" = "id")) 

# analysis set of words per topic, useful for Shiny apps (if needed)

tidy_beta <- data.frame(topic = as.integer(stringr::str_replace_all(rownames(final_model$phi), "t_", "")),
                       final_model$phi,
                       stringsAsFactors = FALSE) %>% 
  gather(term, beta, -topic) %>% 
  mutate(term = str_replace_all(term, "\\.", " ")) %>% 
  tibble::as_tibble()
