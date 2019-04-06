echo 'orig'
sh evaluate.sh original
echo 'orig deep'
sh evaluate.sh original_deep
echo 'random_fix'
sh evaluate.sh random_fix
echo 'random_fix_deep'
sh evaluate.sh random_fix_deep
echo 'random_unfix'
sh evaluate.sh random_unfix
echo 'random_unfix_deep'
sh evaluate.sh random_unfix_deep
echo 'sym'
sh evaluate.sh sym
echo 'sym_deep'
sh evaluate.sh sym_deep
echo 'EMFSaA'
sh evaluate.sh EMFSaA
