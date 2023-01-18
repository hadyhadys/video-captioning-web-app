from modules.tokenization import BertTokenizer


class Postprocessing:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def postprocess(self, pred_list):
        """
        Convert the given list of word indices into a human-readable sentence
        """
        all_result_lists = []
        for re_idx, re_list in enumerate(pred_list):
            decode_text_list = self.tokenizer.convert_ids_to_tokens(re_list)
            if "[SEP]" in decode_text_list:
                SEP_index = decode_text_list.index("[SEP]")
                decode_text_list = decode_text_list[:SEP_index]
            if "[PAD]" in decode_text_list:
                PAD_index = decode_text_list.index("[PAD]")
                decode_text_list = decode_text_list[:PAD_index]
            decode_text = ' '.join(decode_text_list)
            decode_text = decode_text.replace(" ##", "").strip("##").strip()
            all_result_lists.append(decode_text)
        return all_result_lists