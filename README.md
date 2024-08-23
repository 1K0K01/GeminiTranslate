# GeminiTranslate: Free Python Translation API using Google Gemini

This is a translation API written in Python, deployed using 'Vercel' and powered by Google's Gemini API, which utilizes Google's generative AI technology.

## Usage

언젠가 기록 예정

## Notes

- Make sure you can access the Google API, otherwise you may need to use a proxy.
- The API is limited to 60 times per minute, you can [apply for a higher limit here](https://ai.google.dev/docs/increase_quota), or set the maximum number of requests per second to 1 in the translation custom options.
- Prompt injection may exist in the translation result.
- Gemini API is not allow to talk about OpenAI😑
- Recommended to use the maximum number of requests per second in the custom options: 1, the maximum number of paragraphs per request: 20, to avoid exceeding the limit.
