from modules.tokenization import BertTokenizer
from modules.beam import Beam


import torch

class Inference:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def infer(self, model, video, video_mask, args):
        """
        Return the caption, given the video representation and video masking using the deployed model.
        The beam search decoding is used to generate the caption given the word probabilities.
        """
        with torch.no_grad():
            visual_output = model.get_visual_output(video, video_mask)

            # -- Repeat data for beam search
            n_bm = 5 # beam_size
            device = visual_output.device
            n_inst, len_v, v_h = visual_output.size()

            decoder = model.decoder_caption

            # Note: shaped first, then decoder need the parameter shaped=True
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            visual_output_rpt = visual_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
            video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)

            # Prepare beams
            inst_dec_beams = [Beam(n_bm, device=device, tokenizer=self.tokenizer) for _ in range(n_inst)]
            # Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = self._get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            # Decode
            for len_dec_seq in range(1, args.max_words + 1):
                active_inst_idx_list = self._beam_decode_step(decoder, inst_dec_beams,
                                                        len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                        (visual_output_rpt, video_mask_rpt))

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                (visual_output_rpt, video_mask_rpt), \
                inst_idx_to_position_map = self._collate_active_info((visual_output_rpt, video_mask_rpt),
                                                               inst_idx_to_position_map, active_inst_idx_list, n_bm, device)

            batch_hyp, batch_scores = self._collect_hypothesis_and_scores(inst_dec_beams, 1)
            pred_list = [batch_hyp[i][0] for i in range(n_inst)]
        return pred_list

    def _get_inst_idx_to_tensor_position_map(self, inst_idx_list):
        """
        Indicate the position of an instance in a tensor.
        """
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}


    def _beam_decode_step(self, decoder, inst_dec_beams, len_dec_seq,
                     inst_idx_to_position_map, n_bm, device, input_tuples, decoder_length=None):
        """
        Decode and update beam status, and then return active beam idx
        """
        assert isinstance(input_tuples, tuple)

        def _prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def _predict_word(next_decoder_ids, n_active_inst, n_bm, device, input_tuples):
            visual_output_rpt, video_mask_rpt = input_tuples
            next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)

            dec_output = decoder(visual_output_rpt, video_mask_rpt, next_decoder_ids, next_decoder_mask, shaped=True, get_logits=True)
            dec_output = dec_output[:, -1, :]
            word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
            word_prob = word_prob.view(n_active_inst, n_bm, -1)
            return word_prob

        def _collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                if decoder_length is None:
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                else:
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)
        dec_seq = _prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        word_prob = _predict_word(dec_seq, n_active_inst, n_bm, device, input_tuples)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = _collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                            decoder_length=decoder_length)

        return active_inst_idx_list

    def _collate_active_info(self, input_tuples, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
        assert isinstance(input_tuples, tuple)
        visual_output_rpt, video_mask_rpt = input_tuples

        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

        active_visual_output_rpt = self._collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
        active_video_mask_rpt = self._collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
        active_inst_idx_to_position_map = self._get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return (active_visual_output_rpt, active_video_mask_rpt), \
               active_inst_idx_to_position_map


    def _collect_active_part(self, beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        """
        Collect tensor parts associated to active instances.
        """
        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def _collect_hypothesis_and_scores(self, inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores